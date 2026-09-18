#!/usr/bin/env python3
"""
morph_degree_ari.py
===================

Degree-vs-ARI overlay for a user-supplied *morph* (an ordered sequence of point
clouds), with selectable Mapper filter functions.

For each morph step and each selected filter, the script builds a Mapper graph,
then for every pair of adjacent cover elements computes on their overlap slab:

    cross-degree   xdeg_i = 2 * ||N_i||_0 / (J_i + L_i)
    agreement      ARI_i  = adjusted_rand_score(labels from element i,
                                                labels from element i+1)

where N_i is the contingency table between the two clusterings restricted to the
slab. Both are functionals of the same table: degree counts occupied cells,
ARI weights them by C(n,2). The plot overlays one series per filter.

--------------------------------------------------------------------------
INPUT FORMATS  (Configured via INPUT_PATH variable)

  *.npz        either a key 'steps' holding an array of shape (S, n, d),
               or keys step_0, step_1, ... each of shape (n_s, d)
  *.npy        array of shape (S, n, d)
  *.csv        a 'step' column plus numeric coordinate columns
  directory/   one .csv or .npy per step, loaded in sorted filename order

Steps may have different point counts; they need the same dimension.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from collections import defaultdict

import numpy as np

warnings.filterwarnings("ignore")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:                                    # plotting optional
    plt = None

from scipy.spatial.distance import pdist, squareform
from scipy.sparse.linalg import eigsh
from scipy import stats
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

_EPS = 1e-12


# =========================================================================
# USER CONFIGURATION FOR INPUTS
# =========================================================================
# Set your morph input path here (e.g., "morph.npz", "steps/", or "data.csv")
INPUT_PATH = "/Users/sebastianlemberger/Documents/Mapper Experiments/twin_arcs_point_cloud.csv"

# Comma-separated filter names or list of filters to use
SELECTED_FILTERS = "xcoord,knn_k1,dtm_k10,random, ecc_p1,gauss_0.1,dm_eig0"


# Clustering method: "kmeans", "xmeans", or "dbscan".
CLUSTERER_TYPE = "dbscan"
DBSCAN_EPS = 0.1           # Neighbourhood radius in the input coordinate units.
DBSCAN_MIN_SAMPLES = 5    # Includes the point itself; k is unused by DBSCAN.
# Noise (label -1) is omitted from Mapper. Slab ARI uses only observations
# assigned to a cluster in BOTH adjacent bins, with at least min_slab points.
# DBSCAN is deterministic for fixed input order; random filters still vary by seed.


# =========================================================================
# 1. Clusterers
# =========================================================================
class FixedKMeans:
    """k-means with a fixed number of clusters."""

    def __init__(self, k, random_state=0, n_init=5):
        self.k, self.random_state, self.n_init = k, random_state, n_init

    def fit(self, X):
        k = int(min(self.k, len(X)))
        if k <= 1:
            self.labels_ = np.zeros(len(X), dtype=int)
            return self
        self.labels_ = KMeans(n_clusters=k, n_init=self.n_init,
                              random_state=self.random_state).fit_predict(X)
        return self


class XMeans:
    """Pelleg-Moore X-Means: k-means with BIC-accepted recursive 2-way splits."""

    def __init__(self, k_init=1, k_max=30, random_state=0, n_init=5):
        self.k_init, self.k_max = k_init, k_max
        self.random_state, self.n_init = random_state, n_init

    @staticmethod
    def _bic(X, labels, centers):
        R, d = X.shape
        K = centers.shape[0]
        if R <= K:
            return -np.inf
        resid = X - centers[labels]
        var = float(np.einsum("ij,ij->", resid, resid)) / (R - K)
        if var <= _EPS:
            return -np.inf
        cnt = np.bincount(labels, minlength=K).astype(float)
        cnt = cnt[cnt > 0]
        ll = np.sum(-cnt / 2 * np.log(2 * np.pi)
                    - (cnt * d) / 2 * np.log(var)
                    - (cnt - K) / 2
                    + cnt * np.log(cnt) - cnt * np.log(R))
        p = (K - 1) + d * K + 1
        return float(ll - (p / 2) * np.log(R))

    def _km(self, X, k, rng, init=None):
        k = int(min(k, len(X)))
        if k <= 1:
            return X.mean(0, keepdims=True), np.zeros(len(X), dtype=int)
        kw = dict(n_clusters=k, random_state=rng.randint(0, 2**31 - 1))
        km = (KMeans(init=np.asarray(init, float), n_init=1, **kw) if init is not None
              else KMeans(init="k-means++", n_init=self.n_init, **kw))
        km.fit(X)
        return km.cluster_centers_, km.labels_

    def fit(self, X):
        X = np.asarray(X, float)
        if len(X) < 4:
            self.labels_ = np.zeros(len(X), dtype=int)
            return self
        rng = np.random.RandomState(self.random_state)
        centers, labels = self._km(X, max(1, self.k_init), rng)
        for _ in range(25):
            if centers.shape[0] >= self.k_max:
                break
            new, changed = [], False
            for j in range(centers.shape[0]):
                pts = X[labels == j]
                if len(pts) < 4 or centers.shape[0] + len(new) >= self.k_max:
                    new.append(centers[j]); continue
                base = self._bic(pts, np.zeros(len(pts), int), centers[j].reshape(1, -1))
                cc, cl = self._km(pts, 2, rng)
                if len(np.unique(cl)) < 2 or not np.isfinite(base):
                    new.append(centers[j]); continue
                if self._bic(pts, cl, cc) > base:
                    new.extend(list(cc)); changed = True
                else:
                    new.append(centers[j])
            if not changed:
                break
            centers, labels = self._km(X, len(new), rng, init=np.asarray(new))
        uniq = np.unique(labels)
        remap = {o: n for n, o in enumerate(uniq)}
        self.labels_ = np.array([remap[v] for v in labels], dtype=int)
        return self


# =========================================================================
# 2. Filters
# =========================================================================
def _nn_dists(X, k, include_self=False):
    q = min((k if include_self else k + 1), len(X))
    d, _ = NearestNeighbors(n_neighbors=q).fit(X).kneighbors(X)
    return d if include_self else d[:, 1:]


def _ecc(X, p):
    D = squareform(pdist(X))
    return D.max(1) if np.isinf(p) else (np.mean(D ** p, axis=1)) ** (1.0 / p)


def _gauss(X, sigma):
    D = squareform(pdist(X))
    n, d = X.shape
    return np.exp(-(D ** 2) / (2 * sigma ** 2)).sum(1) / (n * (np.sqrt(2 * np.pi) * sigma) ** d)


def _laplacian(X, nbrs=10, which=1):
    A = NearestNeighbors(n_neighbors=min(nbrs, len(X) - 1)).fit(X).kneighbors_graph(X)
    A = ((A + A.T) > 0).astype(float).toarray()
    deg = A.sum(1); deg[deg == 0] = 1.0
    dinv = 1.0 / np.sqrt(deg)
    L = np.eye(len(deg)) - A * dinv[:, None] * dinv[None, :]
    vals, vecs = eigsh(L, k=which + 1, sigma=-1e-5, which="LM")
    return vecs[:, np.argsort(vals)[which]]


def _dm_eig(X, which=0):
    M = squareform(pdist(X))
    M = M - M.mean(0, keepdims=True) - M.mean(1, keepdims=True) + M.mean()
    vals, vecs = np.linalg.eigh(M)
    return vecs[:, np.argsort(-np.abs(vals))[which]]


def _pca1(X):
    Z = X - X.mean(0)
    return Z @ np.linalg.svd(Z, full_matrices=False)[2][0]


FILTERS = {
    "knn_k1":     (lambda X, s: _nn_dists(X, 1)[:, 0],            "distance to nearest other point"),
    "knn_k10":    (lambda X, s: _nn_dists(X, 10)[:, 9],           "distance to 10th nearest neighbour"),
    "dtm_k10":    (lambda X, s: np.sqrt(np.mean(_nn_dists(X, 10) ** 2, 1)),
                                                                  "distance-to-measure, k=10"),
    "ecc_p1":     (lambda X, s: _ecc(X, 1.0),                     "eccentricity, exponent 1"),
    "ecc_p2":     (lambda X, s: _ecc(X, 2.0),                     "eccentricity, exponent 2"),
    "ecc_inf":    (lambda X, s: _ecc(X, np.inf),                  "eccentricity, exponent inf"),
    "gauss_0.1":  (lambda X, s: _gauss(X, 0.1),                   "Gaussian density, sigma=0.1"),
    "gauss_0.3":  (lambda X, s: _gauss(X, 0.3),                   "Gaussian density, sigma=0.3"),
    "laplacian":  (lambda X, s: _laplacian(X),                    "Fiedler vector of the kNN Laplacian"),
    "dm_eig0":    (lambda X, s: _dm_eig(X, 0),                    "leading distance-matrix eigenvector"),
    "dm_eig1":    (lambda X, s: _dm_eig(X, 1),                    "2nd distance-matrix eigenvector"),
    "xcoord":     (lambda X, s: X[:, 0],                          "first coordinate"),
    "pca1":       (lambda X, s: _pca1(X),                         "first principal component"),
    "random":     (lambda X, s: np.random.RandomState((s + 900000011) % (2**31 - 1)).uniform(0, 1, len(X)),
                                                                  "uniform random (null lens)"),
}


# =========================================================================
# 3. Cover and Mapper
# =========================================================================
def _intervals(lo, hi, n, p):
    L = (hi - lo) / (n - (n - 1) * p) if n > 1 else (hi - lo)
    return [(lo + i * L * (1 - p), lo + i * L * (1 - p) + L) for i in range(n)]


def uniform_cover(lens, n, p):
    f = np.asarray(lens).ravel()
    return [np.flatnonzero((f >= a) & (f <= b)) for a, b in _intervals(f.min(), f.max(), n, p)]


def quantile_cover(lens, n, p):
    f = np.asarray(lens).ravel()
    N = len(f)
    order = np.argsort(f, kind="mergesort")
    u = np.empty(N); u[order] = (np.arange(N) + 0.5) / N
    return [np.flatnonzero((u >= a) & (u <= b)) for a, b in _intervals(0.0, 1.0, n, p)]


def mapper_slabs(X, lens, cover_fn, make_clusterer, n_cubes, overlap, min_slab=30):
    nodes = {}
    for ci, idx in enumerate(cover_fn(lens, n_cubes, overlap)):
        if len(idx) == 0:
            continue
        lab = make_clusterer().fit(X[idx]).labels_
        for c in np.unique(lab):
            if c >= 0:
                nodes[(ci, int(c))] = idx[lab == c]

    owner = defaultdict(list)
    for key, idx in nodes.items():
        for pt in idx.tolist():
            owner[pt].append(key)
    edges = set()
    for ks in owner.values():
        for a in range(len(ks)):
            for b in range(a + 1, len(ks)):
                edges.add(tuple(sorted((ks[a], ks[b]))))
    V = len(nodes)
    graph_deg = 2.0 * len(edges) / V if V else 0.0

    byc = defaultdict(list)
    for key in nodes:
        byc[key[0]].append(key)
    out, pair_masses = [], []
    cubes = sorted(byc)
    for i in range(len(cubes) - 1):
        ca, cb = cubes[i], cubes[i + 1]
        if cb != ca + 1:
            continue
        A, B = byc[ca], byc[cb]
        sa = {q: set(nodes[q].tolist()) for q in A}
        sb = {q: set(nodes[q].tolist()) for q in B}
        S = set().union(*sa.values()) & set().union(*sb.values())
        if len(S) < min_slab:
            continue
        iA = {p: j for j, q in enumerate(A) for p in sa[q]}
        iB = {p: l for l, q in enumerate(B) for p in sb[q]}
        S = sorted(S)
        la = np.array([iA[p] for p in S]); lb = np.array([iB[p] for p in S])
        T = np.zeros((len(A), len(B))); np.add.at(T, (la, lb), 1.0)
        occ = T[T > 0]
        K = int(len(occ))
        vbar = len(S) / K if K else np.nan
        cv2 = float(np.var(occ) / vbar ** 2) if K else np.nan
        pair_masses.extend(occ.tolist())
        out.append(dict(slab=ca, n_slab=len(S), J=len(A), L=len(B), cells=K,
                        xdeg=2.0 * K / (len(A) + len(B)),
                        cv2=cv2,
                        ari=float(adjusted_rand_score(la, lb))))
    return out, graph_deg, V, np.asarray(pair_masses, float)


def plot_clouds(steps, path, proj="pca", color_vals=None, color_name=None,
                max_points=4000, title=None, seed=0):
    if plt is None:
        return False
    S = len(steps)
    ncols = min(6, max(1, int(np.ceil(np.sqrt(S)))))
    nrows = int(np.ceil(S / ncols))

    allX = np.vstack(steps)
    d = allX.shape[1]
    if d == 1:
        rng = np.random.RandomState(seed)
        project = lambda Z: np.column_stack([Z[:, 0], rng.uniform(-1, 1, len(Z))])
        labels = ("value", "jitter")
    elif proj == "pca" and d > 2:
        mu = allX.mean(0)
        Vt = np.linalg.svd(allX - mu, full_matrices=False)[2][:2]
        project = lambda Z: (Z - mu) @ Vt.T
        labels = ("PC1", "PC2")
    else:
        project = lambda Z: Z[:, :2]
        labels = ("dim 1", "dim 2")

    P_all = project(allX)
    pad = 0.05 * max(np.ptp(P_all[:, 0]), np.ptp(P_all[:, 1]), 1e-9)
    xlim = (P_all[:, 0].min() - pad, P_all[:, 0].max() + pad)
    ylim = (P_all[:, 1].min() - pad, P_all[:, 1].max() + pad)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 3.1 * nrows),
                             squeeze=False)
    rng = np.random.RandomState(seed)
    for i, ax in enumerate(axes.ravel()):
        if i >= S:
            ax.axis("off"); continue
        X = steps[i]
        idx = (rng.choice(len(X), max_points, replace=False)
               if len(X) > max_points else np.arange(len(X)))
        Q = project(X[idx])
        if color_vals is not None:
            c = np.asarray(color_vals[i]).ravel()[idx]
            ax.scatter(Q[:, 0], Q[:, 1], s=3, c=c, cmap="viridis", alpha=.7, lw=0)
        else:
            ax.scatter(Q[:, 0], Q[:, 1], s=3, c="#34495e", alpha=.5, lw=0)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"step {i}   n={len(X)}", fontsize=9)
    sub = f"projection: {labels[0]} / {labels[1]}"
    if color_name:
        sub += f"   |   colour = {color_name}"
    fig.suptitle((title or "Morph steps") + f"\n{sub}", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close(fig)
    return True


# =========================================================================
# 4. Input
# =========================================================================
def load_morph(path):
    if os.path.isdir(path):
        files = sorted(f for f in os.listdir(path) if f.endswith((".csv", ".npy")))
        if not files:
            raise ValueError(f"no .csv or .npy files in {path}")
        steps = []
        for f in files:
            fp = os.path.join(path, f)
            steps.append(np.load(fp) if f.endswith(".npy")
                         else np.genfromtxt(fp, delimiter=",", skip_header=0))
        return [np.atleast_2d(s).astype(float) for s in steps]

    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        z = np.load(path)
        if "steps" in z:
            arr = z["steps"]
            return [np.asarray(arr[i], float) for i in range(arr.shape[0])]
        keys = sorted((k for k in z.files if k.startswith("step")),
                      key=lambda s: int("".join(c for c in s if c.isdigit()) or 0))
        if not keys:
            raise ValueError("npz needs a 'steps' array or step_0, step_1, ... keys")
        return [np.asarray(z[k], float) for k in keys]
    if ext == ".npy":
        arr = np.load(path)
        if arr.ndim != 3:
            raise ValueError(".npy must have shape (steps, n, d)")
        return [np.asarray(arr[i], float) for i in range(arr.shape[0])]
    if ext == ".csv":
        import csv as _csv
        with open(path) as fh:
            rows = list(_csv.DictReader(fh))
        if not rows or "step" not in rows[0]:
            raise ValueError("csv needs a 'step' column plus coordinate columns")
        cols = [c for c in rows[0] if c != "step"]
        byst = defaultdict(list)
        for r in rows:
            byst[int(float(r["step"]))].append([float(r[c]) for c in cols])
        return [np.asarray(byst[s], float) for s in sorted(byst)]
    raise ValueError(f"unsupported input: {path}")


# =========================================================================
# 5. Main
# =========================================================================
def main():
    # Configuration parameters (formerly CLI arguments)
    input_path = INPUT_PATH
    filters_str = SELECTED_FILTERS
    clusterer_type = CLUSTERER_TYPE.strip().lower()
    k_val = 10
    n_cubes = 10
    overlap = 0.49
    cover_type = "quantile"
    seeds = 3
    level = "slab"
    out_prefix = "morph_degree_ari"
    no_clouds = False
    cloud_proj = "pca"
    cloud_color = "auto"
    title = None

    if clusterer_type not in {"kmeans", "xmeans", "dbscan"}:
        print("Error: CLUSTERER_TYPE must be kmeans, xmeans, or dbscan.", file=sys.stderr)
        return 1
    if clusterer_type == "dbscan":
        if not np.isfinite(DBSCAN_EPS) or DBSCAN_EPS <= 0:
            print("Error: DBSCAN_EPS must be finite and positive.", file=sys.stderr)
            return 1
        if (isinstance(DBSCAN_MIN_SAMPLES, bool)
                or not isinstance(DBSCAN_MIN_SAMPLES, (int, np.integer))
                or DBSCAN_MIN_SAMPLES < 1):
            print("Error: DBSCAN_MIN_SAMPLES must be a positive integer.", file=sys.stderr)
            return 1
        clusterer_label = (f"dbscan, eps={DBSCAN_EPS:g}, "
                           f"min_samples={DBSCAN_MIN_SAMPLES}")
    else:
        clusterer_label = f"{clusterer_type}, k={k_val}"
    print(f"clusterer: {clusterer_label}")

    if not input_path:
        print("Error: INPUT_PATH must be specified in the script configuration.", file=sys.stderr)
        return 1

    sel = [f.strip() for f in filters_str.split(",") if f.strip()]
    unknown = [f for f in sel if f not in FILTERS]
    if unknown:
        print(f"Error: unknown filter(s): {', '.join(unknown)}.", file=sys.stderr)
        return 1

    steps = load_morph(input_path)
    print(f"loaded {len(steps)} morph steps, "
          f"{steps[0].shape[1]}-dimensional, "
          f"{min(len(s) for s in steps)}-{max(len(s) for s in steps)} points per step")

    cover_fn = quantile_cover if cover_type == "quantile" else uniform_cover
    if overlap >= 0.5:
        print("warning: overlap >= 0.5 lets non-consecutive intervals meet; "
              "slab pairing assumes they do not.", file=sys.stderr)

    rows, cv2_rows = [], []
    for si, X in enumerate(steps):
        for f in sel:
            for seed in range(seeds):
                lens = np.asarray(FILTERS[f][0](X, seed), float).reshape(-1, 1)
                if clusterer_type == "kmeans":
                    mk = lambda sd=seed: FixedKMeans(k_val, random_state=sd)
                elif clusterer_type == "xmeans":
                    mk = lambda sd=seed: XMeans(1, k_val, random_state=sd)
                else:
                    mk = lambda: DBSCAN(eps=DBSCAN_EPS,
                                        min_samples=DBSCAN_MIN_SAMPLES)
                slabs, gdeg, V, masses = mapper_slabs(X, lens, cover_fn, mk,
                                                      n_cubes, overlap)
                cv2_pooled = (float(np.var(masses) / (masses.mean() ** 2))
                              if len(masses) > 1 and masses.mean() > 0 else np.nan)
                cv2_slabmean = float(np.nanmean([r["cv2"] for r in slabs])) if slabs else np.nan
                cv2_rows.append(dict(step=si, filter=f, seed=seed,
                                     cv2_pooled=cv2_pooled,
                                     cv2_slabmean=cv2_slabmean,
                                     n_pairs=len(masses)))
                if not slabs:
                    continue
                if level == "slab":
                    for r in slabs:
                        rows.append(dict(step=si, filter=f, seed=seed,
                                         graph_degree=gdeg, n_nodes=V, **r))
                else:
                    rows.append(dict(step=si, filter=f, seed=seed,
                                     graph_degree=gdeg, n_nodes=V,
                                     slab=-1, n_slab=float(np.mean([r["n_slab"] for r in slabs])),
                                     J=-1, L=-1,
                                     cells=float(np.mean([r["cells"] for r in slabs])),
                                     xdeg=float(np.mean([r["xdeg"] for r in slabs])),
                                     cv2=float(np.nanmean([r["cv2"] for r in slabs])),
                                     ari=float(np.mean([r["ari"] for r in slabs]))))
        print(f"  step {si+1}/{len(steps)} done", flush=True)

    if not rows:
        print("No usable slabs produced (at least 30 shared clustered points required). "
              "Try cover_type='quantile' or a larger overlap. For DBSCAN, inspect "
              "noise and consider increasing DBSCAN_EPS or lowering "
              "DBSCAN_MIN_SAMPLES; for k-means/X-means, try a smaller k_val.",
              file=sys.stderr)
        return 1

    import csv as _csv
    cpath = f"{out_prefix}.csv"
    with open(cpath, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {cpath} ({len(rows)} rows)")

    if cv2_rows:
        by = defaultdict(dict)
        for r in cv2_rows:
            by[(r["step"], r["filter"])].setdefault("p", []).append(r["cv2_pooled"])
            by[(r["step"], r["filter"])].setdefault("s", []).append(r["cv2_slabmean"])
        wide_path = f"{out_prefix}_cv2.csv"
        with open(wide_path, "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["step"] + sel)
            for st in sorted({r["step"] for r in cv2_rows}, reverse=True):
                row = [st]
                for f in sel:
                    vals = by.get((st, f), {}).get("p", [])
                    vals = [v for v in vals if np.isfinite(v)]
                    row.append(f"{np.mean(vals):.6f}" if vals else "")
                w.writerow(row)
        print(f"wrote {wide_path} (CV^2 over connected node pairs; "
              f"rows descending by morph step)")

        wide2 = f"{out_prefix}_cv2_slabmean.csv"
        with open(wide2, "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["step"] + sel)
            for st in sorted({r["step"] for r in cv2_rows}, reverse=True):
                row = [st]
                for f in sel:
                    vals = [v for v in by.get((st, f), {}).get("s", []) if np.isfinite(v)]
                    row.append(f"{np.mean(vals):.6f}" if vals else "")
                w.writerow(row)
        print(f"wrote {wide2} (mean of per-slab CV^2, for comparison)")

    if not no_clouds:
        cname = (sel[0] if cloud_color == "auto" else cloud_color)
        cvals = None
        if cname and cname != "none":
            if cname not in FILTERS:
                print(f"warning: unknown cloud-color '{cname}', drawing uncoloured",
                      file=sys.stderr)
                cname = None
            else:
                cvals = [np.asarray(FILTERS[cname][0](X, 0), float) for X in steps]
        else:
            cname = None
        gpath = f"{out_prefix}_clouds.png"
        if plot_clouds(steps, gpath, proj=cloud_proj, color_vals=cvals,
                       color_name=cname, title=title or "Morph steps"):
            print(f"wrote {gpath}")
        else:
            print("matplotlib unavailable; skipped the cloud figure.")

    print(f"\n{'filter':12s} {'n':>6} {'degree range':>16} {'ARI range':>18} "
          f"{'r(deg,ARI)':>11} {'mean CV^2':>10}")
    for f in sel:
        v = [r for r in rows if r["filter"] == f]
        if len(v) < 3:
            print(f"{f:12s} {len(v):>6}  (too few points)"); continue
        d = np.array([r["xdeg"] for r in v]); s = np.array([r["ari"] for r in v])
        rr = stats.pearsonr(d, s)[0] if d.std() > 1e-12 and s.std() > 1e-12 else np.nan
        cv = [r["cv2_pooled"] for r in cv2_rows
              if r["filter"] == f and np.isfinite(r["cv2_pooled"])]
        print(f"{f:12s} {len(v):>6} {f'{d.min():.2f} - {d.max():.2f}':>16} "
              f"{f'{s.min():.3f} - {s.max():.3f}':>18} {rr:>+11.3f} "
              f"{np.mean(cv) if cv else float('nan'):>10.4f}")

    if plt is None:
        print("\nmatplotlib unavailable; skipped the figure.")
        return 0

    fig, ax = plt.subplots(figsize=(9.5, 7))
    cmap = plt.get_cmap("tab10")
    for i, f in enumerate(sel):
        v = [r for r in rows if r["filter"] == f]
        if len(v) < 3:
            continue
        d = np.array([r["xdeg"] for r in v]); s = np.array([r["ari"] for r in v])
        st = np.array([r["step"] for r in v], float)
        col = cmap(i % 10)
        ax.scatter(d, s, s=26, color=col, alpha=.28 + .5 * (st / max(st.max(), 1)),
                   edgecolors="none")
        if d.std() > 1e-12:
            res = stats.linregress(d, s)
            xs = np.linspace(d.min(), d.max(), 20)
            ax.plot(xs, res.intercept + res.slope * xs, color=col, lw=2.2, ls="--",
                    label=f"{f}   r={res.rvalue:+.3f}")
    base = 2.0 - 2.0 / n_cubes
    ax.axhline(1.0, color="gray", ls=":", lw=1)
    ax.axvline(1.0, color="gray", ls=":", lw=1)
    ax.annotate("perfect agreement\n(bijection: xdeg=1, ARI=1)", xy=(1.0, 1.0),
                xytext=(1.05, 0.93), fontsize=8, color="gray")
    ax.set_xlabel("slab cross-degree  $\\bar\\delta_i = 2\\,\\|N_i\\|_0/(J_i+L_i)$", fontsize=11)
    ax.set_ylabel("slab ARI", fontsize=11)
    ax.set_title(title or
                 f"Degree vs ARI across morph  ({clusterer_label}, "
                 f"{cover_type} cover, {n_cubes} intervals, overlap {overlap:g})\n"
                 f"colour = filter, opacity = morph step",
                 fontsize=12)
    ax.grid(alpha=.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    ppath = f"{out_prefix}.png"
    plt.savefig(ppath, dpi=140)
    print(f"wrote {ppath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())