import sys
import numpy as np
import math
from sklearn.metrics import silhouette_score


#Reuse helpers & constants from your symnmf.py
from symnmf import (
    load_points,
    parse_int_like_k,
    init_H,
    handle_error,
    EPS, MAX_ITER, BETA, DEN_EPS,  #EPS=1e-4, MAX_ITER=300
)

#C extension
import symnmf_c as cmod

#HW1 K-means helpers (pure helpers only)
def initialize_centroids(points, K):
    """Select the first K points as initial centroids."""
    return [points[i][:] for i in range(K)]

def assign_clusters(points, centroids):
    """Assign each point to the closest centroid."""
    cluster_indices = []
    for p in points:
        min_dist = float('inf')
        min_index = -1
        for j, c in enumerate(centroids):
            d = dist(p, c)
            if d < min_dist:
                min_dist = d
                min_index = j
        cluster_indices.append(min_index)
    return cluster_indices

def update_centroids(points, cluster_indices, K, dim, old_centroids=None):
    """Compute mean of clusters; keep old centroid if cluster is empty."""
    new_centroids = [[0.0] * dim for _ in range(K)]
    counts = [0] * K

    for i, cluster in enumerate(cluster_indices):
        for d in range(dim):
            new_centroids[cluster][d] += points[i][d]
        counts[cluster] += 1

    for j in range(K):
        if counts[j] == 0 and old_centroids is not None:
            new_centroids[j] = old_centroids[j][:]
            continue
        if counts[j] > 0:
            for d in range(dim):
                new_centroids[j][d] /= counts[j]

    return new_centroids

def dist(p, q):
    """Euclidean distance between two points."""
    return math.sqrt(sum((pi - qi) ** 2 for pi, qi in zip(p, q)))

def has_converged(old_centroids, new_centroids, epsilon):
    """Return True if all centroids moved less than epsilon."""
    for c_old, c_new in zip(old_centroids, new_centroids):
        if dist(c_old, c_new) >= epsilon:
            return False
    return True

#End of kmeans helpers

#Clustering algorithm
def assign_clusters_from_H(H):
    """Convert a SymNMF matrix H (n×k) into hard labels."""
    H_np = np.asarray(H, dtype=float)
    if H_np.ndim != 2 or H_np.shape[1] <= 0:
        handle_error()
    return np.argmax(H_np, axis=1)

def run_kmeans_with_hw1_helpers(points_ll, k, max_iter, epsilon):
    """Run K-means using HW1 helpers on in-memory data."""
    n = len(points_ll)
    if not (1 < k < n):
        handle_error()

    dim = len(points_ll[0])
    centroids = initialize_centroids(points_ll, k)

    for _ in range(max_iter):  # we will call with MAX_ITER
        cluster_indices = assign_clusters(points_ll, centroids)
        # pass old centroids to keep empty-cluster behavior
        new_centroids = update_centroids(points_ll, cluster_indices, k, dim, old_centroids=centroids)
        if has_converged(centroids, new_centroids, epsilon):  # we will call with EPS
            centroids = new_centroids
            break
        centroids = new_centroids

    labels = assign_clusters(points_ll, centroids)
    return np.asarray(labels, dtype=int)

def main():
    try:
        # Expect exactly: analysis.py k file_name
        if len(sys.argv) != 3:
            handle_error()

        k_raw = sys.argv[1]
        file_name = sys.argv[2]

        # Parse & validate k
        k = parse_int_like_k(k_raw)
        if k is None:
            handle_error()

        #Load data (list of lists) + numpy for silhouette
        X_ll = load_points(file_name)
        X_np = np.asarray(X_ll, dtype=float)
        n = X_np.shape[0]
        if not (1 < k < n):
            handle_error()

        #---- SymNMF path (EPS=1e-4, MAX_ITER=300) ----
        W_ll = cmod.norm(X_ll)
        H0 = init_H(W_ll, k)
        Hf = cmod.symnmf(H0.tolist(), W_ll, EPS, MAX_ITER, BETA, DEN_EPS)
        y_nmf = assign_clusters_from_H(Hf)
        nmf_score = float(silhouette_score(X_np, y_nmf, metric="euclidean"))

        #---- KMeans path (use same fixed EPS/MAX_ITER) ----
        y_km = run_kmeans_with_hw1_helpers(X_ll, k, max_iter=MAX_ITER, epsilon=EPS)
        km_score = float(silhouette_score(X_np, y_km, metric="euclidean"))

        # Output
        print(f"nmf: {nmf_score:.4f}")
        print(f"kmeans: {km_score:.4f}")

    except Exception:
        handle_error()

if __name__ == "__main__":
    main()
