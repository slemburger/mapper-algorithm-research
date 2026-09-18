import json
import csv
import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix

rng = np.random.default_rng()

def mean_degree(X, k, n_cubes=10, gain=0.5):
    f = X[:, 0]
    lo, hi = f.min(), f.max()
    w = (hi - lo)/n_cubes
    half = 0.5*w/(1 - gain)
    rows, cols = [], []
    node = 0
    for c in lo + w*(np.arange(n_cubes) + 0.5):
        idx = np.where((f >= c - half) & (f <= c + half))[0]
        if len(idx) < max(k, 5):
            continue
        kk = min(k, len(idx))
        lab = KMeans(n_clusters=kk, n_init=1).fit_predict(X[idx])
        for j in range(kk):
            s = idx[lab == j]
            if len(s):
                rows.extend([node]*len(s))
                cols.extend(s.tolist())
                node += 1
    if node < 2:
        return 0.0
    M = csr_matrix((np.ones(len(rows), np.int8), (rows, cols)),
                   shape=(node, X.shape[0]), dtype=np.int8)
    A = (M @ M.T).astype(bool).toarray()
    np.fill_diagonal(A, False)
    return float(A.sum(1).mean())

KS = list(range(1, 50))
DIMS = list(range(1, 100)) + [150, 200, 250, 500]
N, REPS = 10000, 30

csv_rows = []

for d in DIMS:
    for k in KS:
        v = [mean_degree(rng.random((N, d)), k) for _ in range(REPS)]
        
        mean_val = float(np.mean(v))
        std_val = float(np.std(v))
        
        # Calculate Coefficient of Variation (CV), safeguard against division by zero
        cv_val = float(std_val / mean_val) if mean_val > 0 else 0.0
        
        csv_rows.append({
            "Dimension": d,
            "K": k,
            "Mean_Degree": round(mean_val, 2),
            "CV_Degree": round(cv_val, 4)
        })
        print(f"Dim {d}, K {k} -> Mean: {mean_val:.2f}, CV: {cv_val:.4f}", flush=True)

# Export to CSV
with open("degree_stats.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["Dimension", "K", "Mean_Degree", "CV_Degree"])
    writer.writeheader()
    writer.writerows(csv_rows)

print("Successfully exported all data to degree_stats.csv!")