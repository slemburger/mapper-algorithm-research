#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
mapper.py  --  The Mapper algorithm (Singh, Memoli, Carlsson 2007) on a CSV
               point cloud, with separate PNG exports of (i) the point cloud
               and (ii) the Mapper graph.
================================================================================

FORMAL SPECIFICATION
--------------------
Let  X = {x_1, ..., x_n} subset R^d  be a finite point cloud (the rows of the
CSV), equipped with the Euclidean metric  rho(x,y) = ||x - y||_2.

1. FILTER (lens).  Fix a continuous map
        f : X --> R,
   and set  I = [a, b]  with  a = min_i f(x_i),  b = max_i f(x_i).

2. COVER.  Fix N in N (N_BINS) and an overlap fraction g in [0,1) (OVERLAP).
   Let  R = b - a  and  L = R / (N - (N-1)g).  Define
        U_i = [ a + i*L*(1-g),  a + i*L*(1-g) + L ],      i = 0, ..., N-1.
   Then  U = {U_i}_{i=0}^{N-1}  is a finite cover of I by closed intervals of
   common length L, consecutive intervals overlapping in a set of length gL,
   and  U_{N-1} terminates exactly at b, since
        a + (N-1)L(1-g) + L = a + L(N - (N-1)g) = a + R = b.

3. PULLBACK.  X_i := f^{-1}(U_i) subset X.  The family {X_i} covers X.

4. REFINEMENT BY CLUSTERING.  Apply a clustering functor C to each X_i,
   obtaining a partition (modulo discarded noise)
        X_i = C_{i,1} \sqcup ... \sqcup C_{i,k_i}  (\sqcup N_i, the noise set).
   The collection  V = { C_{i,j} }  is the *refined pullback cover* of X.

5. NERVE.  The Mapper graph is the 1-skeleton of the nerve N(V):
        Vertices:  V = { C_{i,j} }
        Edges:     { C_{i,j}, C_{i',j'} }  in E  iff  C_{i,j} \cap C_{i',j'} != 0
   with edge weight  w(u,v) = | C_u \cap C_v |  (cardinality of the overlap).
   Since each C_{i,j} lies in a single pullback X_i and the clustering of X_i
   is a partition, edges only occur between distinct intervals; in particular
   G is a simple graph with no self-loops.

OPTIONAL ANNOTATIONS RENDERED IN THE CORNER OF THE GRAPH FIGURE
---------------------------------------------------------------
(a) AVERAGE VERTEX DEGREE.  For the simple undirected graph G = (V, E),
        deg_avg(G) = (1/|V|) sum_{v in V} deg(v) = 2|E| / |V|
    by the handshake lemma  sum_v deg(v) = 2|E|.

(b) PAIRWISE ADJACENT-BIN ADJUSTED RAND INDEX.  For adjacent intervals U_i,
    U_{i+1}, let  S = X_i \cap X_{i+1} = f^{-1}(U_i \cap U_{i+1}).  Let
    pi = C(X_i)|_S  and  sigma = C(X_{i+1})|_S  be the two induced partitions
    of S, with contingency table  n_{ab} = |pi_a \cap sigma_b|,
    a_. = sum_b n_{ab},  b_. = sum_a n_{ab},  m = |S|.  Then

        ARI(pi, sigma) =
            ( sum_{ab} C(n_ab,2) - [sum_a C(a_.,2) sum_b C(b_.,2)] / C(m,2) )
          / ( 1/2 [sum_a C(a_.,2) + sum_b C(b_.,2)]
              - [sum_a C(a_.,2) sum_b C(b_.,2)] / C(m,2) )

    i.e. the Rand index corrected for chance under the generalised
    hypergeometric null model; ARI = 1 iff pi = sigma, and E[ARI] = 0 under
    independence.  This measures the *stability* of the local clustering
    across the overlap of consecutive bins: values near 1 indicate that the
    nerve edges between those bins are geometrically trustworthy.

(c) PAIRWISE ADJACENT-BIN RENYI GAP.  For the K nonempty intersections between
    clusters in adjacent bins, let n_e be their edge masses, m = sum_e n_e,
    and p_e = n_e/m.  The order-0 and order-2 Renyi entropies are
        H_0 = log K,                 H_2 = -log sum_e p_e^2.
    The Renyi gap and effective edge count are
        Delta = H_0 - H_2 = log(K sum_e p_e^2),
        K_eff = exp(H_2) = 1 / sum_e p_e^2.
    Delta = 0 exactly when the positive edge masses are uniform, and grows as
    overlap mass concentrates on a smaller subset of the observed edges.  All
    logarithms are natural.  K = 0 is reported as undefined; K = 1 has gap 0.

CLUSTERING FUNCTORS AVAILABLE
-----------------------------
  'dbscan'  Density-based (Ester et al. 1996).  Parameters (eps, minPts);
            eps may be estimated per bin as the q-quantile of the empirical
            distribution of k-th nearest-neighbour distances,
                eps = Q_q( { d_k(x) : x in X_i } ),  k = minPts - 1.
            Noise points (label -1) are handled per DBSCAN_NOISE_POLICY.

  'kmeans'  Lloyd's algorithm minimising  sum_{j} sum_{x in C_j} ||x - mu_j||^2
            with a fixed number of centroids k (clamped to |X_i|).

  'xmeans'  Pelleg & Moore (2000).  Model selection by the Bayesian
            Information Criterion under a spherical-Gaussian mixture with a
            single shared variance
                sigma^2 = (1/(d(R-K))) sum_i ||x_i - mu_{c(i)}||^2,
            log-likelihood
                l = sum_n [ -R_n/2 log(2 pi) - (R_n d)/2 log(sigma^2)
                            - (R_n - K)/2 + R_n log R_n - R_n log R ],
            free parameters  p = (K-1) + dK + 1,  and
                BIC(M_K) = l - (p/2) log R.
            Each cluster is bisected by 2-means and the split is accepted iff
            BIC(M_2) > BIC(M_1) locally, iterating until K = XMEANS_K_MAX.

EVERYTHING IS CONFIGURED IN THE CONFIGURATION BLOCK BELOW.  There are no
command-line arguments.

Requires: numpy, pandas, scikit-learn, matplotlib  (networkx optional).
================================================================================
"""

# ==============================================================================
# ============================ CONFIGURATION BLOCK =============================
# ========== Edit everything here; no command-line arguments are used. ==========
# ==============================================================================

# ------------------------------- I/O ------------------------------------------
# Relative paths are resolved first against the current working directory and,
# failing that, against the directory containing this script, so the file can be
# run from anywhere.  OUTPUT_DIR is created next to the input CSV.
INPUT_CSV          = "/Users/sebastianlemberger/Documents/Mapper Experiments/renyi_mass_shift_point_cloud.csv"
CSV_HAS_HEADER     = True                # True -> first row is a header
USE_COLUMNS        = None                # None = all numeric columns; else a
                                         # list of names (if header) or integer
                                         # indices, e.g. ["x","y","z"] / [0,1,2]
EXCLUDE_COLUMNS    = []                 # columns kept out of the METRIC but
                                         # still readable as a lens, e.g.
                                         # ["filter_value", "true_blob"].
                                         # Class labels and precomputed filter
                                         # values are NOT coordinates of X.
DROP_NON_NUMERIC   = True                # silently drop non-numeric columns
DROP_NA_ROWS       = True                # drop rows containing NaN
SUBSAMPLE_N        = None                # None or an int: random subsample size

OUTPUT_DIR         = "mapper_output"
POINT_CLOUD_IMAGE  = "point_cloud.png"   # image 1: the point cloud
MAPPER_GRAPH_IMAGE = "mapper_graph.png"  # image 2: the Mapper graph
FIGURE_DPI         = 200
FIGSIZE            = (14.0, 7.0)          # inches
SHOW_PLOTS         = False               # also open interactive windows

# --------------------------- PREPROCESSING ------------------------------------
RESCALE            = "none"              # "none" | "standardize" | "minmax"
RANDOM_SEED        = 0

# ------------------------------ FILTER f --------------------------------------
# "coordinate"      f(x) = x_k                     (k = FILTER_COORDINATE)
# "norm"            f(x) = ||x - xbar||_2
# "pca1" / "pca2"   f(x) = <x - xbar, v_1>  /  <x - xbar, v_2>   (principal axes)
# "eccentricity"    f(x) = ( (1/n) sum_j ||x - x_j||^p )^(1/p)   (p = ECC_P;
#                   p = inf gives f(x) = max_j ||x - x_j||)
# "density"         f(x) = (1/n) sum_j exp(-||x - x_j||^2 / eps) (Gaussian KDE)
# "knn_distance"    f(x) = (1/k) sum over the k nearest neighbours of ||x - .||
# "column"          f(x) = value of FILTER_COLUMN for x (a lens precomputed in
#                   the CSV; the column is automatically removed from X)
# "custom"          f = CUSTOM_FILTER  (see below)
FILTER_FUNCTION    = "column"
FILTER_COORDINATE  = 0                   # used by "coordinate"
FILTER_COLUMN      = "time"      # used by "column"
ECC_P              = 2.0                 # used by "eccentricity"; float("inf") ok
DENSITY_EPSILON    = "auto"              # "auto" (median pairwise distance^2) or float
KNN_K              = 5                   # used by "knn_distance"
# CUSTOM_FILTER must map an (n,d) array to an (n,) array; np is in scope at call:
CUSTOM_FILTER      = lambda X: X[:, 0] ** 2 + X[:, 1] ** 2

# ------------------------------- COVER U --------------------------------------
N_BINS             = 10                  # N: number of intervals (resolution)
OVERLAP            = 0.4               # g in [0,1): overlap fraction (gain)
COVER_MODE         = "uniform"           # "uniform"  -> intervals of equal LENGTH
                                         # "balanced" -> intervals of equal MASS
                                         #    (pullback of a uniform cover of
                                         #    [0,1] under the empirical CDF of f;
                                         #    use this when f is multimodal and
                                         #    uniform bins come out empty)

# ---------------------------- CLUSTERING C ------------------------------------
CLUSTERER          = "kmeans"            # "dbscan" | "kmeans" | "xmeans"

# -- DBSCAN --
DBSCAN_EPS         = 0.1              # "auto" or a positive float
DBSCAN_MIN_SAMPLES = 5            # "auto" -> minPts = max(2d, ceil(ln|X_i|)),
                                         # which grows with the bin population;
                                         # a fixed small value shatters the tails
                                         # of large bins into micro-clusters
DBSCAN_EPS_QUANTILE = 0.95               # q in (0,1] for the "auto" heuristic
DBSCAN_NOISE_POLICY = "discard"          # "discard" | "singleton" | "own_cluster"

# Clusters smaller than this are dropped from the refined cover (an int, or a
# float in (0,1) read as a fraction of the bin population).  0 disables it.
MIN_CLUSTER_SIZE   = 0

# -- KMeans --
KMEANS_K           = 3                   # k per bin (clamped to bin cardinality)
KMEANS_N_INIT      = 50

# -- XMeans --
XMEANS_K_MIN       = 1
XMEANS_K_MAX       = 8
XMEANS_N_INIT      = 50

# Bins with fewer than this many points are emitted as a single cluster
# (set to 0 to disable and let the clusterer handle them).
MIN_POINTS_SINGLE_CLUSTER = 0

# ---------------------------- ANNOTATION --------------------------------------
# "none" | "avg_degree" | "ari" | "renyi" | "both" | "all"
# "both" retains its historical meaning: degree + ARI.  "all" adds Renyi.
ANNOTATION         = "avg_degree"
ANNOTATION_PLACEMENT = "title"           # "title"  -> compact one/two-line header
                                         #             (deg=..  V=..  E=..  b1=..)
                                         # "corner" -> verbose box inside the axes
ANNOTATION_CORNER  = "upper left"        # "upper left"|"upper right"|
                                         # "lower left"|"lower right"
ANNOTATION_FONTSIZE = 9
TITLE_PREFIX       = ""                  # e.g. "step 0" -> "step 0   deg=2.40"
TITLE_SHOW_PARAMS  = False               # append f, N, g, C to the title line
ARI_MAX_LINES      = 12                  # truncate the per-pair ARI listing
ARI_NOISE_POLICY   = "singleton"         # how noise enters the partitions used
                                         # for ARI: "singleton"|"one_cluster"|"exclude"
RENYI_MAX_LINES    = 12                  # truncate the per-pair Renyi listing
RENYI_GAP_CSV      = "renyi_gap.csv"     # written in OUTPUT_DIR; None disables

# ------------------------------ DRAWING ---------------------------------------
GRAPH_STYLE        = "minimal"           # "minimal" -> no axes, frame, ticks or
                                         #              quantitative axes; equal aspect
                                         # "axes"    -> full axes + colorbar
GRAPH_LAYOUT       = "filter"            # "spring" | "kamada_kawai" | "filter"
                                         # ("spring"/"kamada_kawai" need networkx)
NODE_SIZE_RANGE    = (15.0, 140.0)       # scatter areas for min/max cluster size
NODE_EDGE_COLOR    = "black"
NODE_EDGE_WIDTH    = 0.6
EDGE_WIDTH_RANGE   = (0.8, 1.8)
EDGE_COLOR         = "0.55"
EDGE_ALPHA         = 0.9
NODE_COLOR_BY      = "bin"               # "bin" (discrete cover bin) or "filter"
COLORMAP           = "viridis"           # palette for node colours
SHOW_BIN_COLORBAR  = True                # numbered bin key below the Mapper graph
BIN_LABEL_BASE     = 1                   # display bins as 1..N (use 0 for 0..N-1)
SHOW_COLORBAR      = True               # continuous bar for NODE_COLOR_BY="filter";
                                         # forced off in the "minimal" style
SHOW_NODE_LABELS   = False
SHOW_SIZE_LEGEND   = False               # legend mapping marker area to |C|
SPRING_K           = None                # optimal edge length; None -> use
                                         # SPRING_K_SCALE / sqrt(|V_c|)
SPRING_K_SCALE     = 2.0                 # >1 spreads dense components out
SPRING_ITERATIONS  = 200
SPRING_USE_WEIGHTS = False               # True -> spring stiffness prop. to
                                         # |C_u \cap C_v| (collapses hubs)
PACK_COMPONENTS    = True                # lay out each connected component
                                         # separately and tile them in a grid
CLOUD_STYLE        = "minimal"           # "minimal" | "axes" (same convention
                                         # as GRAPH_STYLE, applied to image 1)
POINT_CLOUD_COLOR_BY = "filter"          # "filter" | "bin" | "none"
POINT_SIZE         = 10.0

# ==============================================================================
# ========================= END OF CONFIGURATION BLOCK =========================
# ==============================================================================

import os
import math
import networkx
import itertools
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
if not SHOW_PLOTS:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm
from matplotlib.lines import Line2D

from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, pairwise_distances
from sklearn.preprocessing import MinMaxScaler, StandardScaler

try:
    import networkx as nx
    _HAVE_NX = True
except Exception:  # pragma: no cover
    _HAVE_NX = False


# ==============================================================================
# 1.  DATA LOADING
# ==============================================================================

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:                       # interactive session
    SCRIPT_DIR = os.getcwd()


def resolve_path(path: str) -> str:
    """Absolute paths are used as given; relative paths are looked for first in
    the working directory, then beside this script.  This lets the file be run
    as `python /some/where/mapper.py` from any directory."""
    if os.path.isabs(path):
        return path
    here = os.path.abspath(path)
    if os.path.exists(here):
        return here
    beside = os.path.join(SCRIPT_DIR, path)
    return beside if os.path.exists(beside) else here


def load_point_cloud(path: str) -> Tuple[np.ndarray, List[str], Dict[str, np.ndarray]]:
    r"""Load X = {x_1,...,x_n} \subset R^d from a CSV file.

    Columns listed in EXCLUDE_COLUMNS (and the lens column, when
    FILTER_FUNCTION = "column") are withheld from X: they are metadata, not
    coordinates, and including them would change the metric rho and hence the
    clustering, the nerve, and every downstream statistic.

    Returns
    -------
    X      : (n, d) float64 array of coordinates
    names  : list of d column labels
    extras : dict mapping each withheld column name to its (n,) value vector
    """
    resolved = resolve_path(path)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(
            f"Input CSV not found: {path!r} (looked in {os.getcwd()!r} and "
            f"{SCRIPT_DIR!r}). Set INPUT_CSV to an absolute path in the "
            f"configuration block, or run make_demo_csv()."
        )
    header = 0 if CSV_HAS_HEADER else None
    df = pd.read_csv(resolved, header=header)
    if not CSV_HAS_HEADER:
        df.columns = [f"x{j}" for j in range(df.shape[1])]

    if USE_COLUMNS is not None:
        if all(isinstance(c, (int, np.integer)) for c in USE_COLUMNS):
            df = df.iloc[:, list(USE_COLUMNS)]
        else:
            df = df.loc[:, list(USE_COLUMNS)]

    if DROP_NON_NUMERIC:
        numeric = df.select_dtypes(include=[np.number])
        dropped = [c for c in df.columns if c not in numeric.columns]
        if dropped:
            print(f"[data] dropping non-numeric columns: {dropped}")
        df = numeric

    if DROP_NA_ROWS:
        before = len(df)
        df = df.dropna(axis=0)
        if len(df) < before:
            print(f"[data] dropped {before - len(df)} rows containing NaN")

    if SUBSAMPLE_N is not None and SUBSAMPLE_N < len(df):
        rng = np.random.default_rng(RANDOM_SEED)
        idx = np.sort(rng.choice(len(df), size=int(SUBSAMPLE_N), replace=False))
        df = df.iloc[idx]
        print(f"[data] subsampled to n = {len(df)}")

    # ---- withhold metadata columns from the metric -------------------------
    withheld: List[str] = []
    for c in EXCLUDE_COLUMNS:
        col = df.columns[c] if isinstance(c, (int, np.integer)) else c
        if col in df.columns:
            withheld.append(col)
        else:
            print(f"[warn] EXCLUDE_COLUMNS entry {c!r} is not a column; ignored.")
    if FILTER_FUNCTION.lower() == "column" and FILTER_COLUMN in df.columns:
        if FILTER_COLUMN not in withheld:
            withheld.append(FILTER_COLUMN)

    extras = {c: df[c].to_numpy(dtype=np.float64) for c in withheld}
    if withheld:
        print(f"[data] withheld from the metric (available as a lens): {withheld}")
    coords = df.drop(columns=withheld)

    X = np.ascontiguousarray(coords.to_numpy(dtype=np.float64))
    names = [str(c) for c in coords.columns]

    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 1:
        raise ValueError(
            f"Degenerate point cloud of shape {X.shape}: no coordinate columns "
            f"remain after exclusions."
        )

    # ---- heuristic sanity check on the remaining coordinates ---------------
    for j, nm in enumerate(names):
        col = X[:, j]
        u = np.unique(col)
        if u.size <= 20 and np.allclose(u, np.round(u)):
            print(f"[warn] coordinate {nm!r} takes only {u.size} distinct integer "
                  f"values; if it is a class label or an index, add it to "
                  f"EXCLUDE_COLUMNS - it is currently deforming the metric.")

    return X, names, extras


def rescale(X: np.ndarray) -> np.ndarray:
    """Apply the configured affine renormalisation to each coordinate."""
    if RESCALE == "none":
        return X
    if RESCALE == "standardize":     # z_j = (x_j - mu_j) / sigma_j
        return StandardScaler().fit_transform(X)
    if RESCALE == "minmax":          # z_j = (x_j - min_j)/(max_j - min_j)
        return MinMaxScaler().fit_transform(X)
    raise ValueError(f"Unknown RESCALE = {RESCALE!r}")


# ==============================================================================
# 2.  FILTER FUNCTIONS  f : X -> R
# ==============================================================================

def _pairwise(X: np.ndarray) -> np.ndarray:
    """Full Euclidean distance matrix D_{ij} = ||x_i - x_j||_2 (O(n^2) memory)."""
    n = X.shape[0]
    if n > 8000:
        print(f"[warn] computing a {n}x{n} distance matrix; this is O(n^2).")
    return pairwise_distances(X, metric="euclidean")


def compute_filter(X: np.ndarray,
                   extras: Optional[Dict[str, np.ndarray]] = None
                   ) -> Tuple[np.ndarray, str]:
    r"""Evaluate the configured lens f on X; return (f(X), LaTeX-ish label)."""
    name = FILTER_FUNCTION.lower()
    n, d = X.shape
    extras = extras or {}

    if name == "column":
        if FILTER_COLUMN not in extras:
            raise ValueError(
                f"FILTER_FUNCTION='column' but {FILTER_COLUMN!r} was not found "
                f"in the CSV (available withheld columns: {sorted(extras)})."
            )
        v = np.asarray(extras[FILTER_COLUMN], dtype=float)
        if v.shape[0] != n:
            raise ValueError("Lens column length does not match n.")
        return v, rf"$f = ${FILTER_COLUMN}"

    if name == "coordinate":
        k = int(FILTER_COORDINATE)
        if not (0 <= k < d):
            raise ValueError(f"FILTER_COORDINATE={k} out of range for d={d}.")
        return X[:, k].astype(float), rf"$f(x)=x_{{{k}}}$"

    if name == "norm":
        c = X.mean(axis=0)
        return np.linalg.norm(X - c, axis=1), r"$f(x)=\|x-\bar{x}\|_2$"

    if name in ("pca1", "pca2"):
        comp = 1 if name == "pca1" else 2
        if d < comp:
            raise ValueError(f"{name} requires d >= {comp}, got d = {d}.")
        p = PCA(n_components=comp, random_state=RANDOM_SEED).fit(X)
        scores = p.transform(X)[:, comp - 1]
        ev = p.explained_variance_ratio_[comp - 1]
        return scores, rf"$f(x)=\langle x-\bar{{x}},v_{{{comp}}}\rangle$ ({ev:.1%} var.)"

    if name == "eccentricity":
        D = _pairwise(X)
        p = float(ECC_P)
        if math.isinf(p):
            return D.max(axis=1), r"$f(x)=\max_j\|x-x_j\|$"
        with np.errstate(over="ignore"):
            val = (np.mean(D ** p, axis=1)) ** (1.0 / p)
        return val, rf"$f(x)=\left(n^{{-1}}\sum_j\|x-x_j\|^{{{p:g}}}\right)^{{1/{p:g}}}$"

    if name == "density":
        D = _pairwise(X)
        if DENSITY_EPSILON == "auto":
            med = np.median(D[D > 0]) if np.any(D > 0) else 1.0
            eps = float(med ** 2)
        else:
            eps = float(DENSITY_EPSILON)
        val = np.exp(-(D ** 2) / eps).mean(axis=1)
        return val, rf"$f(x)=n^{{-1}}\sum_j e^{{-\|x-x_j\|^2/{eps:.3g}}}$"

    if name == "knn_distance":
        D = _pairwise(X)
        k = max(1, min(int(KNN_K), n - 1))
        Ds = np.sort(D, axis=1)[:, 1:k + 1]     # exclude the self-distance 0
        return Ds.mean(axis=1), rf"$f(x)=k^{{-1}}\sum_{{j\in N_{{{k}}}(x)}}\|x-x_j\|$"

    if name == "custom":
        val = np.asarray(CUSTOM_FILTER(X), dtype=float).ravel()
        if val.shape[0] != n:
            raise ValueError("CUSTOM_FILTER must return one scalar per point.")
        return val, r"$f = $ CUSTOM_FILTER"

    raise ValueError(f"Unknown FILTER_FUNCTION = {FILTER_FUNCTION!r}")


# ==============================================================================
# 3.  COVER OF THE IMAGE  I = [a,b]
# ==============================================================================

@dataclass
class Cover:
    lows: np.ndarray            # (N,) left endpoints
    highs: np.ndarray           # (N,) right endpoints
    length: float               # L
    overlap: float              # g

    def __len__(self) -> int:
        return len(self.lows)


def build_cover(f: np.ndarray, n_bins: int, overlap: float) -> Cover:
    r"""Cover of [min f, max f] by N overlapping closed intervals.

    COVER_MODE = "uniform":  intervals of equal length
        L = R/(N-(N-1)g),  U_i = [a + iL(1-g),  a + iL(1-g) + L].

    COVER_MODE = "balanced": intervals of equal empirical mass.  Let F_n be the
    empirical CDF of f and let {[l_i, r_i]} be the uniform cover of [0,1] with
    the same N and g.  Put
        U_i = [ F_n^{-1}(l_i),  F_n^{-1}(r_i) ],
    the pullback of the uniform cover under F_n.  Then |f^{-1}(U_i)| ~ n/(N-(N-1)g)
    for every i, so no pullback can be empty (up to ties in f).  This is the
    correct choice when f is multimodal or heavy-tailed: a uniform cover then
    places intervals in gaps of the image, producing empty bins and a spuriously
    disconnected nerve.
    """
    if n_bins < 1:
        raise ValueError("N_BINS must be >= 1.")
    if not (0.0 <= overlap < 1.0):
        raise ValueError("OVERLAP must lie in [0,1).")
    a, b = float(f.min()), float(f.max())
    R = b - a
    if R <= 0:                      # constant filter: degenerate cover
        R = 1.0
        b = a + R

    mode = COVER_MODE.lower()
    if mode == "balanced":
        Lq = 1.0 / (n_bins - (n_bins - 1) * overlap)
        lq = np.arange(n_bins) * Lq * (1.0 - overlap)
        rq = np.minimum(lq + Lq, 1.0)
        lows = np.quantile(f, lq)
        highs = np.quantile(f, rq)
        L = float(np.mean(highs - lows))
    elif mode == "uniform":
        L = R / (n_bins - (n_bins - 1) * overlap)
        lows = a + np.arange(n_bins) * L * (1.0 - overlap)
        highs = lows + L
    else:
        raise ValueError(f"Unknown COVER_MODE = {COVER_MODE!r}")

    highs[-1] = b + 1e-12           # guarantee closure at the right endpoint
    lows[0] = a - 1e-12
    return Cover(lows=lows, highs=highs, length=L, overlap=overlap)


def pullback(f: np.ndarray, cover: Cover) -> List[np.ndarray]:
    r"""X_i = f^{-1}(U_i), returned as arrays of indices into X."""
    return [np.flatnonzero((f >= lo) & (f <= hi))
            for lo, hi in zip(cover.lows, cover.highs)]


# ==============================================================================
# 4.  CLUSTERING FUNCTORS
# ==============================================================================

def _resolve_min_samples(m: int, d: int) -> int:
    r"""minPts for DBSCAN.  With DBSCAN_MIN_SAMPLES = "auto",
        minPts = max(2d, ceil(ln m)),
    combining the Sander et al. rule minPts ~ 2d with logarithmic growth in the
    bin population m.  This matters because the "auto" eps is a quantile of the
    k-th nearest-neighbour distribution with k = minPts - 1: as m grows at fixed
    k, d_k(x) -> 0 and eps collapses, so a fixed small minPts makes DBSCAN carve
    the low-density tails of a large bin into spurious micro-clusters."""
    if isinstance(DBSCAN_MIN_SAMPLES, str) and DBSCAN_MIN_SAMPLES.lower() == "auto":
        return max(2 * d, int(math.ceil(math.log(max(m, 3)))), 3)
    return max(2, int(DBSCAN_MIN_SAMPLES))


def _auto_eps(Xs: np.ndarray, min_samples: int, q: float) -> float:
    r"""eps = Q_q({ d_k(x) }), k = min_samples - 1, d_k = k-th NN distance."""
    m = Xs.shape[0]
    k = max(1, min(int(min_samples), m) - 1)
    D = pairwise_distances(Xs)
    dk = np.sort(D, axis=1)[:, k]
    eps = float(np.quantile(dk, q))
    if not np.isfinite(eps) or eps <= 0.0:
        pos = D[D > 0]
        eps = float(pos.min()) if pos.size else 1e-9
    return eps


def _spherical_bic(Xs: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> float:
    r"""BIC of a K-component spherical Gaussian mixture with shared variance
    (Pelleg & Moore, 2000).  Larger is better."""
    R, d = Xs.shape
    K = centers.shape[0]
    if R <= K:
        return -np.inf
    counts = np.zeros(K)
    ss = 0.0
    for k in range(K):
        mask = labels == k
        counts[k] = int(mask.sum())
        if counts[k] > 0:
            diff = Xs[mask] - centers[k]
            ss += float(np.sum(diff * diff))
    var = ss / (d * (R - K))
    if not np.isfinite(var) or var <= 0.0:
        var = 1e-12
    ll = 0.0
    for k in range(K):
        Rn = counts[k]
        if Rn <= 0:
            continue
        ll += (-0.5 * Rn * math.log(2.0 * math.pi)
               - 0.5 * Rn * d * math.log(var)
               - 0.5 * (Rn - K)
               + Rn * math.log(Rn)
               - Rn * math.log(R))
    p = (K - 1) + d * K + 1
    return ll - 0.5 * p * math.log(R)


def _xmeans(Xs: np.ndarray, k_min: int, k_max: int, seed: int) -> np.ndarray:
    """X-means: BIC-driven recursive bisection between k_min and k_max."""
    m = Xs.shape[0]
    k_min = max(1, min(int(k_min), m))
    k_max = max(k_min, min(int(k_max), m))
    km = KMeans(n_clusters=k_min, n_init=XMEANS_N_INIT, random_state=seed).fit(Xs)
    labels, centers = km.labels_, km.cluster_centers_
    K = centers.shape[0]

    while K < k_max:
        new_centers: List[np.ndarray] = []
        split_happened = False
        for k in range(K):
            pts = Xs[labels == k]
            budget_left = k_max - (len(new_centers) + (K - k))
            if pts.shape[0] < 4 or budget_left < 1:
                new_centers.append(centers[k])
                continue
            km2 = KMeans(n_clusters=2, n_init=XMEANS_N_INIT,
                         random_state=seed).fit(pts)
            if len(np.unique(km2.labels_)) < 2:
                new_centers.append(centers[k])
                continue
            bic1 = _spherical_bic(pts, np.zeros(pts.shape[0], dtype=int),
                                  pts.mean(axis=0)[None, :])
            bic2 = _spherical_bic(pts, km2.labels_, km2.cluster_centers_)
            if bic2 > bic1:
                new_centers.extend(list(km2.cluster_centers_))
                split_happened = True
            else:
                new_centers.append(centers[k])
        if not split_happened:
            break
        init = np.asarray(new_centers, dtype=float)
        K = init.shape[0]
        km = KMeans(n_clusters=K, init=init, n_init=1, random_state=seed).fit(Xs)
        labels, centers = km.labels_, km.cluster_centers_
        K = centers.shape[0]
    return labels


def cluster_subset(Xs: np.ndarray, seed: int) -> np.ndarray:
    r"""Apply the configured clustering functor C to a pullback X_i.

    Returns an (m,) integer label vector; label -1 denotes DBSCAN noise.
    """
    m = Xs.shape[0]
    if m == 0:
        return np.empty(0, dtype=int)
    if m <= max(1, int(MIN_POINTS_SINGLE_CLUSTER)) and MIN_POINTS_SINGLE_CLUSTER > 0:
        return np.zeros(m, dtype=int)

    method = CLUSTERER.lower()
    if method == "dbscan":
        min_samples = _resolve_min_samples(m, Xs.shape[1])
        eps = (_auto_eps(Xs, min_samples, DBSCAN_EPS_QUANTILE)
               if DBSCAN_EPS == "auto" else float(DBSCAN_EPS))
        lab = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(Xs)
        return lab
    if method == "kmeans":
        k = max(1, min(int(KMEANS_K), m))
        return KMeans(n_clusters=k, n_init=KMEANS_N_INIT,
                      random_state=seed).fit_predict(Xs)
    if method == "xmeans":
        return _xmeans(Xs, XMEANS_K_MIN, XMEANS_K_MAX, seed)
    raise ValueError(f"Unknown CLUSTERER = {CLUSTERER!r}")


def labels_to_clusters(idx: np.ndarray, labels: np.ndarray) -> List[np.ndarray]:
    r"""Convert a label vector on X_i into the list of vertex sets C_{i,j},
    applying DBSCAN_NOISE_POLICY to the noise set N_i = {x : label = -1}."""
    clusters: List[np.ndarray] = []
    for lab in sorted(set(labels.tolist()) - {-1}):
        clusters.append(idx[labels == lab])
    noise = idx[labels == -1]
    if noise.size:
        policy = DBSCAN_NOISE_POLICY.lower()
        if policy == "singleton":
            clusters.extend([np.array([i]) for i in noise])
        elif policy == "own_cluster":
            clusters.append(noise)
        # "discard": noise contributes no vertices
    clusters = [c for c in clusters if c.size > 0]
    if MIN_CLUSTER_SIZE:
        thr = (int(math.ceil(MIN_CLUSTER_SIZE * idx.size))
               if 0 < MIN_CLUSTER_SIZE < 1 else int(MIN_CLUSTER_SIZE))
        clusters = [c for c in clusters if c.size >= thr]
    return clusters


# ==============================================================================
# 5.  THE NERVE  N(V)  (1-skeleton)
# ==============================================================================

@dataclass
class Vertex:
    vid: int
    bin_index: int
    members: np.ndarray                 # indices into X
    f_mean: float
    centroid: np.ndarray

    @property
    def size(self) -> int:
        return int(self.members.size)


@dataclass
class MapperGraph:
    vertices: List[Vertex]
    edges: List[Tuple[int, int, int]]   # (u, v, |C_u \cap C_v|)
    cover: Cover
    bin_labels: List[Dict[int, int]] = field(default_factory=list)
    # bin_labels[i] maps a point index -> its cluster label inside bin i
    # (before the noise policy is applied); used for the ARI annotation.

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    def average_degree(self) -> float:
        r"""deg_avg = 2|E|/|V| (handshake lemma); 0 for the empty graph."""
        return 0.0 if self.n_vertices == 0 else 2.0 * self.n_edges / self.n_vertices

    def degree_sequence(self) -> np.ndarray:
        deg = np.zeros(self.n_vertices, dtype=int)
        for u, v, _ in self.edges:
            deg[u] += 1
            deg[v] += 1
        return deg

    def betti_1(self) -> int:
        r"""First Betti number (cycle rank) of the 1-complex G:
            b_1 = rank H_1(G; Z) = |E| - |V| + b_0,
        i.e. the nullity of the boundary map \partial_1 : C_1 -> C_0.  For a
        Mapper graph this counts independent loops, the coarse 1-dimensional
        topology of X detected at this resolution."""
        return self.n_edges - self.n_vertices + self.n_components()

    def n_components(self) -> int:
        """Number of connected components via union-find (= rank of H_0)."""
        parent = list(range(self.n_vertices))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for u, v, _ in self.edges:
            ru, rv = find(u), find(v)
            if ru != rv:
                parent[ru] = rv
        return len({find(i) for i in range(self.n_vertices)})


def build_mapper(X: np.ndarray, f: np.ndarray, cover: Cover) -> MapperGraph:
    """Refined pullback cover + its nerve."""
    pulls = pullback(f, cover)
    vertices: List[Vertex] = []
    bin_labels: List[Dict[int, int]] = []

    for i, idx in enumerate(pulls):
        if idx.size == 0:
            bin_labels.append({})
            print(f"[cover] bin {i:>3}: empty")
            continue
        labels = cluster_subset(X[idx], seed=RANDOM_SEED + i)
        bin_labels.append({int(p): int(l) for p, l in zip(idx, labels)})
        clusters = labels_to_clusters(idx, labels)
        for c in clusters:
            vertices.append(Vertex(vid=len(vertices), bin_index=i, members=np.sort(c),
                                   f_mean=float(f[c].mean()),
                                   centroid=X[c].mean(axis=0)))
        n_noise = int(np.sum(labels == -1))
        print(f"[cover] bin {i:>3}: |X_i| = {idx.size:>5}  clusters = "
              f"{len(clusters):>3}  noise = {n_noise}")

    # Nerve: accumulate |C_u \cap C_v| by inverting the membership relation.
    membership: Dict[int, List[int]] = {}
    for v in vertices:
        for p in v.members.tolist():
            membership.setdefault(p, []).append(v.vid)
    weights: Dict[Tuple[int, int], int] = {}
    for vids in membership.values():
        if len(vids) > 1:
            for u, w in itertools.combinations(sorted(vids), 2):
                weights[(u, w)] = weights.get((u, w), 0) + 1
    edges = [(u, v, w) for (u, v), w in sorted(weights.items())]

    return MapperGraph(vertices=vertices, edges=edges, cover=cover,
                       bin_labels=bin_labels)


# ==============================================================================
# 6.  ANNOTATIONS
# ==============================================================================

def _partition_on(overlap_pts: Sequence[int], lab_map: Dict[int, int],
                  offset: int) -> Optional[np.ndarray]:
    """Extract a label vector on the overlap set, applying ARI_NOISE_POLICY."""
    out: List[int] = []
    nxt = offset
    for p in overlap_pts:
        l = lab_map[p]
        if l == -1:
            pol = ARI_NOISE_POLICY.lower()
            if pol == "singleton":
                out.append(nxt)
                nxt += 1
            elif pol == "one_cluster":
                out.append(-1)
            else:                      # "exclude"
                out.append(-10**9)     # sentinel, filtered by the caller
        else:
            out.append(l)
    return np.asarray(out, dtype=np.int64)


def adjacent_bin_ari(G: MapperGraph) -> List[Tuple[int, int, Optional[float], int]]:
    r"""ARI( C(X_i)|_S , C(X_{i+1})|_S ) on S = X_i \cap X_{i+1},
    for every adjacent pair of intervals.

    Returns a list of tuples (i, i+1, ARI or None, |S|).
    """
    results: List[Tuple[int, int, Optional[float], int]] = []
    for i in range(len(G.bin_labels) - 1):
        A, B = G.bin_labels[i], G.bin_labels[i + 1]
        S = sorted(set(A.keys()) & set(B.keys()))
        if len(S) < 2:
            results.append((i, i + 1, None, len(S)))
            continue
        la = _partition_on(S, A, offset=10_000)
        lb = _partition_on(S, B, offset=20_000)
        if ARI_NOISE_POLICY.lower() == "exclude":
            keep = (la > -10**8) & (lb > -10**8)
            la, lb = la[keep], lb[keep]
        if la.size < 2:
            results.append((i, i + 1, None, int(la.size)))
            continue
        results.append((i, i + 1, float(adjusted_rand_score(la, lb)), int(la.size)))
    return results


@dataclass(frozen=True)
class RenyiGapResult:
    """Order-0/order-2 Renyi diagnostics for one adjacent-bin transition.

    ``mass`` is the sum of the actual Mapper edge weights between the two bins.
    It can be smaller than the raw lens-overlap cardinality when DBSCAN noise or
    clusters removed by ``MIN_CLUSTER_SIZE`` do not become Mapper vertices.
    """
    bin_i: int
    bin_j: int
    mass: int
    K: int
    H0: Optional[float]
    H2: Optional[float]
    gap: Optional[float]
    K_eff: Optional[float]


def adjacent_bin_renyi_gap(G: MapperGraph) -> List[RenyiGapResult]:
    r"""Compute the edge-mass Renyi gap for every adjacent bin pair.

    The positive edge weights between bins i and i+1 are n_1,...,n_K.  With
    p_e = n_e / sum n_e, this returns

        H_0 = log K,
        H_2 = -log sum_e p_e^2,
        gap = H_0 - H_2 = log(K sum_e p_e^2),
        K_eff = exp(H_2) = 1 / sum_e p_e^2.

    The calculation uses the nerve itself, so it automatically respects
    ``DBSCAN_NOISE_POLICY`` and ``MIN_CLUSTER_SIZE``.  Natural logarithms are
    used.  A transition with no Mapper edge has undefined entropies and gap.
    """
    bin_of = {v.vid: v.bin_index for v in G.vertices}
    by_pair: Dict[Tuple[int, int], List[int]] = {
        (i, i + 1): [] for i in range(max(0, len(G.cover) - 1))
    }
    for u, v, weight in G.edges:
        a, b = sorted((bin_of[u], bin_of[v]))
        if b == a + 1 and weight > 0:
            by_pair[(a, b)].append(int(weight))

    results: List[RenyiGapResult] = []
    for i in range(max(0, len(G.cover) - 1)):
        weights = np.asarray(by_pair[(i, i + 1)], dtype=float)
        K = int(weights.size)
        mass = int(weights.sum()) if K else 0
        if K == 0 or mass == 0:
            results.append(RenyiGapResult(
                i, i + 1, mass, K, None, None, None, None
            ))
            continue
        p = weights / float(mass)
        collision = float(np.dot(p, p))
        H0 = float(math.log(K))
        H2 = float(-math.log(collision))
        gap = float(max(0.0, H0 - H2))  # suppress negative roundoff at uniformity
        K_eff = float(1.0 / collision)
        results.append(RenyiGapResult(
            i, i + 1, mass, K, H0, H2, gap, K_eff
        ))
    return results


def write_renyi_gap_csv(G: MapperGraph, path: str) -> None:
    """Write one row per adjacent-bin transition for reproducible analysis."""
    rows = []
    for r in adjacent_bin_renyi_gap(G):
        rows.append({
            "bin_i": r.bin_i,
            "bin_j": r.bin_j,
            "overlap_edge_mass": r.mass,
            "positive_edges_K": r.K,
            "H0_log_K": r.H0,
            "H2_order_2": r.H2,
            "renyi_gap_H0_minus_H2": r.gap,
            "effective_edge_count": r.K_eff,
        })
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[out ] Renyi gap   -> {path}")


def build_annotation_text(G: MapperGraph) -> str:
    """Assemble the corner annotation according to ANNOTATION."""
    mode = ANNOTATION.lower()
    blocks: List[str] = []

    if mode in ("avg_degree", "both", "all"):
        deg = G.degree_sequence()
        blocks.append(
            "Average vertex degree\n"
            f"  $\\bar d = 2|E|/|V| = {G.average_degree():.4f}$\n"
            f"  $|V| = {G.n_vertices}$,  $|E| = {G.n_edges}$,  "
            f"$b_0 = {G.n_components()}$,  $b_1 = {G.betti_1()}$\n"
            f"  $\\deg_{{\\min}} = {int(deg.min()) if deg.size else 0}$,  "
            f"$\\deg_{{\\max}} = {int(deg.max()) if deg.size else 0}$"
        )

    if mode in ("ari", "both", "all"):
        rows = adjacent_bin_ari(G)
        valid = [r[2] for r in rows if r[2] is not None]
        lines = [r"Adjusted Rand index on $X_i\cap X_{i+1}$"]
        shown = rows[:max(1, int(ARI_MAX_LINES))]
        for i, j, val, m in shown:
            lines.append(f"  {i:>2}\u2013{j:<2}  " +
                         (f"ARI = {val:+.3f}   (|S| = {m})" if val is not None
                          else f"undefined (|S| = {m})"))
        if len(rows) > len(shown):
            lines.append(f"  ... {len(rows) - len(shown)} further pairs omitted")
        if valid:
            arr = np.asarray(valid)
            lines.append(f"  mean = {arr.mean():+.3f},  min = {arr.min():+.3f},"
                         f"  max = {arr.max():+.3f}")
        blocks.append("\n".join(lines))

    if mode in ("renyi", "all"):
        rows = adjacent_bin_renyi_gap(G)
        valid = [r.gap for r in rows if r.gap is not None]
        lines = [r"Renyi edge-mass gap $\Delta=H_0-H_2$ (natural log)"]
        shown = rows[:max(1, int(RENYI_MAX_LINES))]
        for r in shown:
            if r.gap is None:
                lines.append(
                    f"  {r.bin_i:>2}–{r.bin_j:<2}  undefined "
                    f"(K = {r.K}, mass = {r.mass})"
                )
            else:
                lines.append(
                    f"  {r.bin_i:>2}–{r.bin_j:<2}  Delta = {r.gap:.3f}   "
                    f"K = {r.K}, K_eff = {r.K_eff:.2f}, mass = {r.mass}"
                )
        if len(rows) > len(shown):
            lines.append(f"  ... {len(rows) - len(shown)} further pairs omitted")
        if valid:
            arr = np.asarray(valid)
            lines.append(
                f"  mean = {arr.mean():.3f},  min = {arr.min():.3f}, "
                f" max = {arr.max():.3f}"
            )
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def build_title_annotation(G: MapperGraph) -> str:
    r"""Compact header form of the diagnostics:
        deg=2.40   V=109 E=131 b_1=27   [ARI mean/min/max]."""
    mode = ANNOTATION.lower()
    lines: List[str] = []
    head = TITLE_PREFIX.strip()

    if mode in ("avg_degree", "both", "all"):
        head = (head + "   " if head else "") + f"deg={G.average_degree():.2f}"
    if head:
        lines.append(head)
    lines.append(f"V={G.n_vertices} E={G.n_edges} "
                 f"$b_0$={G.n_components()} $b_1$={G.betti_1()}")

    if mode in ("ari", "both", "all"):
        vals = [r[2] for r in adjacent_bin_ari(G) if r[2] is not None]
        if vals:
            a = np.asarray(vals)
            lines.append(f"ARI  mean={a.mean():.2f}  min={a.min():.2f}  "
                         f"max={a.max():.2f}")
        else:
            lines.append("ARI  undefined")
    if mode in ("renyi", "all"):
        vals = [r.gap for r in adjacent_bin_renyi_gap(G) if r.gap is not None]
        if vals:
            a = np.asarray(vals)
            lines.append(f"Renyi gap  mean={a.mean():.2f}  min={a.min():.2f}  "
                         f"max={a.max():.2f}")
        else:
            lines.append("Renyi gap  undefined")
    return "\n".join(lines)


def _corner_coords(corner: str) -> Tuple[float, float, str, str]:
    c = corner.lower().strip()
    table = {
        "upper left":  (0.015, 0.985, "left", "top"),
        "upper right": (0.985, 0.985, "right", "top"),
        "lower left":  (0.015, 0.015, "left", "bottom"),
        "lower right": (0.985, 0.015, "right", "bottom"),
    }
    if c not in table:
        raise ValueError(f"Unknown ANNOTATION_CORNER = {corner!r}")
    return table[c]


# ==============================================================================
# 7.  LAYOUT AND DRAWING
# ==============================================================================

def _scale(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return v
    vmin, vmax = v.min(), v.max()
    if vmax - vmin < 1e-12:
        return np.full_like(v, 0.5 * (lo + hi))
    return lo + (hi - lo) * (v - vmin) / (vmax - vmin)


def compute_layout(G: MapperGraph, X: np.ndarray) -> np.ndarray:
    r"""Vertex positions in R^2.

    "filter": the canonical Mapper embedding, x-coordinate = mean filter value
    \bar f(C), y-coordinate = mean second principal score of the members, which
    separates clusters lying in the same interval.
    """
    n = G.n_vertices
    if n == 0:
        return np.zeros((0, 2))
    mode = GRAPH_LAYOUT.lower()

    if mode in ("spring", "kamada_kawai"):
        if not _HAVE_NX:
            print("[warn] networkx unavailable; falling back to the 'filter' layout.")
        else:
            g = nx.Graph()
            g.add_nodes_from(range(n))
            for u, v, w in G.edges:
                g.add_edge(u, v, weight=float(w))
            weight = "weight" if SPRING_USE_WEIGHTS else None

            def _one(sub: "nx.Graph") -> Dict[int, np.ndarray]:
                if sub.number_of_nodes() == 1:
                    return {list(sub.nodes)[0]: np.zeros(2)}
                if mode == "spring":
                    k = (SPRING_K if SPRING_K is not None
                         else SPRING_K_SCALE / math.sqrt(sub.number_of_nodes()))
                    return nx.spring_layout(sub, k=k,
                                            iterations=SPRING_ITERATIONS,
                                            seed=RANDOM_SEED, weight=weight)
                return nx.kamada_kawai_layout(sub, weight=weight)

            if not PACK_COMPONENTS:
                pos = _one(g)
                return np.array([pos[i] for i in range(n)], dtype=float)

            # Lay out each connected component in isolation, normalise it to the
            # unit box scaled by sqrt(|V_c|) (so area ~ vertex count), and tile
            # the components on a square grid.  This prevents a single large
            # component from being crushed towards a point by the repulsion of
            # far-flung isolated vertices.
            comps = sorted(nx.connected_components(g), key=len, reverse=True)
            out = np.zeros((n, 2), dtype=float)
            nmax = max(len(c) for c in comps)
            pad = 0.12
            sides = [math.sqrt(len(c) / nmax) + pad for c in comps]
            # Shelf (first-fit-decreasing) packing into a roughly square region.
            target = math.sqrt(sum(s * s for s in sides)) * 1.3
            x_cur, y_cur, row_h = 0.0, 0.0, 0.0
            for comp, s in zip(comps, sides):
                if x_cur > 0.0 and x_cur + s > target:
                    y_cur -= row_h
                    x_cur, row_h = 0.0, 0.0
                sub = g.subgraph(comp)
                p = _one(sub)
                P = np.array([p[i] for i in sorted(comp)], dtype=float)
                P = P - P.mean(axis=0)
                span = float(np.abs(P).max()) if P.size else 0.0
                if span > 0:
                    P = P / (2.0 * span)                 # into [-0.5, 0.5]^2
                P *= (s - pad)
                P += np.array([x_cur + s / 2.0, y_cur - s / 2.0])
                for k, vid in enumerate(sorted(comp)):
                    out[vid] = P[k]
                x_cur += s
                row_h = max(row_h, s)
            return out

    # "filter" layout
    xs = np.array(
    [v.f_mean for v in G.vertices],
    dtype=float)
    vertical_coordinate = X[:, 1]  # y coordinate
    ys = np.array(
        [
            float(vertical_coordinate[v.members].mean())
            for v in G.vertices
        ],
        dtype=float,)

    # De-collide vertices sharing an interval and a near-identical y value.
    span = float(ys.max() - ys.min()) if ys.max() > ys.min() else 1.0
    by_bin: Dict[int, List[int]] = {}
    for k, v in enumerate(G.vertices):
        by_bin.setdefault(v.bin_index, []).append(k)
    for _, ks in by_bin.items():
        ks_sorted = sorted(ks, key=lambda k: ys[k])
        for r in range(1, len(ks_sorted)):
            prev, cur = ks_sorted[r - 1], ks_sorted[r]
            min_gap = 0.04 * span
            if ys[cur] - ys[prev] < min_gap:
                ys[cur] = ys[prev] + min_gap
    return np.column_stack([xs, ys])


def plot_point_cloud(X: np.ndarray, f: np.ndarray, names: Sequence[str],
                     f_label: str, cover: Cover, path: str) -> None:
    r"""Render X (in R^2, R^3, or projected onto the first two principal axes)."""
    n, d = X.shape
    fig = plt.figure(figsize=FIGSIZE)

    if POINT_CLOUD_COLOR_BY.lower() == "filter":
        c, cmap, cbar_label = f, COLORMAP, f_label
    elif POINT_CLOUD_COLOR_BY.lower() == "bin":
        idx = np.clip(np.digitize(f, cover.lows) - 1, 0, len(cover) - 1)
        c, cmap, cbar_label = idx, COLORMAP, "interval index $i$"
    else:
        c, cmap, cbar_label = None, None, None

    if d == 1:
        ax = fig.add_subplot(111)
        sc = ax.scatter(X[:, 0], np.zeros(n), s=POINT_SIZE, c=c, cmap=cmap)
        ax.set_xlabel(names[0]); ax.set_yticks([])
    elif d == 2:
        ax = fig.add_subplot(111)
        sc = ax.scatter(X[:, 0], X[:, 1], s=POINT_SIZE, c=c, cmap=cmap)
        ax.set_xlabel(names[0]); ax.set_ylabel(names[1]); ax.set_aspect("equal")
    elif d == 3:
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=POINT_SIZE, c=c, cmap=cmap)
        ax.set_xlabel(names[0]); ax.set_ylabel(names[1]); ax.set_zlabel(names[2])
    else:
        p = PCA(n_components=2, random_state=RANDOM_SEED).fit(X)
        Y = p.transform(X)
        ax = fig.add_subplot(111)
        sc = ax.scatter(Y[:, 0], Y[:, 1], s=POINT_SIZE, c=c, cmap=cmap)
        ev = p.explained_variance_ratio_
        ax.set_xlabel(f"PC1 ({ev[0]:.1%})"); ax.set_ylabel(f"PC2 ({ev[1]:.1%})")
        ax.set_aspect("equal")

    if CLOUD_STYLE.lower() == "minimal":
        ax.set_title((TITLE_PREFIX + "   " if TITLE_PREFIX.strip() else "") +
                     f"n={n}  d={d}", fontsize=ANNOTATION_FONTSIZE + 3)
        if d == 3:
            ax.set_axis_off()
        else:
            ax.set_axis_off()
            ax.set_aspect("equal")
    else:
        if c is not None:
            cb = fig.colorbar(sc, ax=ax, shrink=0.85)
            cb.set_label(cbar_label)
        ax.set_title(f"Point cloud  $X \\subset \\mathbb{{R}}^{{{d}}}$,  $n = {n}$")
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    print(f"[out ] point cloud  -> {path}")
    if not SHOW_PLOTS:
        plt.close(fig)


def plot_mapper_graph(G: MapperGraph, X: np.ndarray, f_label: str,
                      path: str) -> None:
    """Render the 1-skeleton of the nerve in the configured style.

    GRAPH_STYLE = "minimal" strips quantitative axes, frame, and ticks and fixes
    an equal aspect ratio.  The discrete bin key remains available below the
    graph when SHOW_BIN_COLORBAR is true.  GRAPH_STYLE = "axes" keeps the full
    quantitative frame.
    """
    minimal = GRAPH_STYLE.lower() == "minimal"
    pos = compute_layout(G, X)
    fig, ax = plt.subplots(figsize=FIGSIZE)

    if G.n_vertices == 0:
        ax.text(0.5, 0.5, "empty Mapper graph", ha="center", va="center",
                transform=ax.transAxes)
    else:
        if G.edges:
            ws = np.array([w for _, _, w in G.edges], dtype=float)
            lws = _scale(ws, *EDGE_WIDTH_RANGE)
            segs = np.array([[[pos[u, 0], pos[u, 1]], [pos[v, 0], pos[v, 1]]]
                             for u, v, _ in G.edges])
            ax.add_collection(LineCollection(
                segs, linewidths=lws, colors=EDGE_COLOR, alpha=EDGE_ALPHA,
                zorder=1, capstyle="round"))
        sizes = _scale(np.array([v.size for v in G.vertices], float),
                       *NODE_SIZE_RANGE)
        color_mode = NODE_COLOR_BY.lower()
        if color_mode == "bin":
            # Use a genuinely discrete palette: every cluster created in the
            # same cover bin receives exactly the same colour.
            n_bins = len(G.cover)
            colors = np.array([v.bin_index for v in G.vertices], float)
            cmap = plt.get_cmap(COLORMAP, n_bins)
            boundaries = np.arange(-0.5, n_bins + 0.5, 1.0)
            norm = BoundaryNorm(boundaries, cmap.N)
            sc = ax.scatter(pos[:, 0], pos[:, 1], s=sizes, c=colors,
                            cmap=cmap, norm=norm,
                            edgecolors=NODE_EDGE_COLOR,
                            linewidths=NODE_EDGE_WIDTH, zorder=2)
            if SHOW_BIN_COLORBAR:
                ticks = np.arange(n_bins)
                cb = fig.colorbar(
                    sc, ax=ax, orientation="horizontal", ticks=ticks,
                    boundaries=boundaries, spacing="uniform", pad=0.07,
                    fraction=0.055, aspect=max(18, 2 * n_bins),
                )
                cb.ax.set_xticklabels(
                    [str(i + BIN_LABEL_BASE) for i in range(n_bins)]
                )
                cb.set_label("Cover bin")
        elif color_mode == "filter":
            colors = np.array([v.f_mean for v in G.vertices], float)
            sc = ax.scatter(pos[:, 0], pos[:, 1], s=sizes, c=colors,
                            cmap=COLORMAP, edgecolors=NODE_EDGE_COLOR,
                            linewidths=NODE_EDGE_WIDTH, zorder=2)
            if SHOW_COLORBAR and not minimal:
                cb = fig.colorbar(sc, ax=ax, shrink=0.85)
                cb.set_label(r"$\bar f(C) = |C|^{-1}\sum_{x \in C} f(x)$")
        else:
            raise ValueError(
                f"Unknown NODE_COLOR_BY={NODE_COLOR_BY!r}; use 'bin' or 'filter'."
            )
        if SHOW_NODE_LABELS:
            for v in G.vertices:
                ax.annotate(f"{v.bin_index}:{v.size}",
                            (pos[v.vid, 0], pos[v.vid, 1]), fontsize=6,
                            ha="center", va="center", zorder=3)
        if SHOW_SIZE_LEGEND:
            handles = [
                Line2D([], [], marker="o", linestyle="none", markerfacecolor="0.7",
                       markeredgecolor="k",
                       markersize=math.sqrt(NODE_SIZE_RANGE[0]),
                       label=f"$|C| = {min(v.size for v in G.vertices)}$"),
                Line2D([], [], marker="o", linestyle="none", markerfacecolor="0.7",
                       markeredgecolor="k",
                       markersize=math.sqrt(NODE_SIZE_RANGE[1]),
                       label=f"$|C| = {max(v.size for v in G.vertices)}$"),
            ]
            opposite = {"upper left": "lower right", "upper right": "lower left",
                        "lower left": "upper right", "lower right": "upper left"}
            loc = opposite.get(ANNOTATION_CORNER.lower(), "lower right")
            ax.legend(handles=handles, loc=loc, frameon=True,
                      fontsize=8, labelspacing=1.4, borderpad=0.8,
                      title="cluster cardinality", title_fontsize=8)

    ax.margins(0.08)
    ax.autoscale_view()

    # ---- frame -------------------------------------------------------------
    if minimal:
        ax.set_axis_off()
        if GRAPH_LAYOUT.lower() == "filter":
            ax.set_aspect("auto")
        else:
            ax.set_aspect("equal")
    else:
        on_filter = GRAPH_LAYOUT.lower() == "filter"
        ax.set_xlabel(r"$\bar f(C)$" if on_filter else "")
        ax.set_ylabel("PC2 (mean over $C$)" if on_filter else "")
        if not on_filter:
            ax.set_xticks([]); ax.set_yticks([])

    # ---- annotation --------------------------------------------------------
    place = ANNOTATION_PLACEMENT.lower()
    params = (f"{f_label},  $N = {len(G.cover)}$,  $g = {G.cover.overlap:.2f}$,  "
              f"C = {CLUSTERER}")

    if place == "title":
        title = build_title_annotation(G)
        if TITLE_SHOW_PARAMS:
            title = (title + "\n" + params) if title else params
        if title:
            ax.set_title(title, fontsize=ANNOTATION_FONTSIZE + 3, linespacing=1.35)
    else:
        ax.set_title(
            f"Mapper graph  $\\mathrm{{sk}}_1\\,\\mathcal{{N}}"
            f"(\\mathcal{{V}})$  —  {params}"
        )
        text = build_annotation_text(G)
        if text:
            x, y, ha, va = _corner_coords(ANNOTATION_CORNER)
            # Enlarge the viewport on the annotated side so the box occupies
            # empty space rather than overlapping the drawing.
            n_lines = text.count("\n") + 1
            pad = min(0.75, 0.045 * n_lines + 0.05)
            y0, y1 = ax.get_ylim()
            h = (y1 - y0) if y1 > y0 else 1.0
            if va == "top":
                ax.set_ylim(y0, y1 + pad * h)
            else:
                ax.set_ylim(y0 - pad * h, y1)
            x0, x1 = ax.get_xlim()
            w = (x1 - x0) if x1 > x0 else 1.0
            if ha == "left":
                ax.set_xlim(x0 - 0.04 * w, x1 + 0.02 * w)
            else:
                ax.set_xlim(x0 - 0.02 * w, x1 + 0.04 * w)
            ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va,
                    fontsize=ANNOTATION_FONTSIZE, family="monospace", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                              edgecolor="0.4", alpha=0.92))

    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"[out ] mapper graph -> {path}")
    if not SHOW_PLOTS:
        plt.close(fig)


# ==============================================================================
# 8.  DEMO DATA (optional convenience)
# ==============================================================================

def make_demo_csv(path: str = "point_cloud.csv", n: int = 800,
                  noise: float = 0.08, seed: int = 0) -> str:
    r"""Write a noisy sample of the circle S^1 \subset R^2; Mapper with a linear
    lens should recover a graph homotopy-equivalent to S^1 (one cycle)."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 2.0 * math.pi, n)
    pts = np.column_stack([np.cos(theta), np.sin(theta)])
    pts += rng.normal(scale=noise, size=pts.shape)
    pd.DataFrame(pts, columns=["x", "y"]).to_csv(path, index=False)
    print(f"[demo] wrote {n} points to {path}")
    return path


# ==============================================================================
# 9.  DRIVER
# ==============================================================================

def main() -> MapperGraph:
    np.random.seed(RANDOM_SEED)

    csv_path = resolve_path(INPUT_CSV)
    out_dir = (OUTPUT_DIR if os.path.isabs(OUTPUT_DIR)
               else os.path.join(os.path.dirname(csv_path), OUTPUT_DIR))
    os.makedirs(out_dir, exist_ok=True)

    X_raw, names, extras = load_point_cloud(INPUT_CSV)
    X = rescale(X_raw)
    print(f"[data] {csv_path}")
    print(f"[data] X in R^{X.shape[1]},  n = {X.shape[0]},  coordinates = {names}")

    f, f_label = compute_filter(X, extras)
    print(f"[lens] {FILTER_FUNCTION}: f(X) in [{f.min():.6g}, {f.max():.6g}]")

    cover = build_cover(f, N_BINS, OVERLAP)
    print(f"[cover] mode = {COVER_MODE},  N = {len(cover)},  g = {OVERLAP},  "
          f"mean L = {cover.length:.6g}")

    G = build_mapper(X, f, cover)
    print(f"[nerve] |V| = {G.n_vertices},  |E| = {G.n_edges},  "
          f"deg_avg = {G.average_degree():.4f},  b0 = {G.n_components()},  "
          f"b1 = {G.betti_1()}")

    n_empty = sum(1 for b in G.bin_labels if not b)
    if n_empty and COVER_MODE.lower() == "uniform":
        print(f"[warn] {n_empty} of {len(cover)} pullbacks are empty: the image "
              f"of f has gaps, so the nerve is disconnected by construction. "
              f"Set COVER_MODE = 'balanced' for equal-mass intervals.")

    if ANNOTATION.lower() in ("ari", "both", "all"):
        for i, j, val, m in adjacent_bin_ari(G):
            s = "undefined" if val is None else f"{val:+.4f}"
            print(f"[ari  ] bins {i}-{j}: ARI = {s}   |S| = {m}")

    if ANNOTATION.lower() in ("renyi", "all"):
        for r in adjacent_bin_renyi_gap(G):
            if r.gap is None:
                print(f"[gap  ] bins {r.bin_i}-{r.bin_j}: undefined   "
                      f"K = {r.K}   mass = {r.mass}")
            else:
                print(f"[gap  ] bins {r.bin_i}-{r.bin_j}: "
                      f"Delta = {r.gap:.4f}   K = {r.K}   "
                      f"K_eff = {r.K_eff:.3f}   mass = {r.mass}")

    pc_path = os.path.join(out_dir, POINT_CLOUD_IMAGE)
    gr_path = os.path.join(out_dir, MAPPER_GRAPH_IMAGE)
    if RENYI_GAP_CSV:
        gap_path = (RENYI_GAP_CSV if os.path.isabs(RENYI_GAP_CSV)
                    else os.path.join(out_dir, RENYI_GAP_CSV))
        write_renyi_gap_csv(G, gap_path)
    plot_point_cloud(X, f, names, f_label, cover, pc_path)
    plot_mapper_graph(G, X, f_label, gr_path)

    if SHOW_PLOTS:
        plt.show()
    return G


if __name__ == "__main__":
    # If the configured CSV does not exist, uncomment the next line to write a
    # noisy circle to it as a smoke test of the whole pipeline:
    #make_demo_csv("users/sebastianlemberger/Downloads/point_cloud.csv")
    main()
