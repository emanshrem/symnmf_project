import sys
import numpy as np
from sklearn.metrics import silhouette_score

#Reuse helpers & constants from your existing symnmf.py (no duplication)
from symnmf import (
    load_points,
    parse_int_like_k,
    init_H,
    handle_error,
    EPS, MAX_ITER, BETA, DEN_EPS, #EPS=1e-4, MAX_ITER=300
)

#Import the C extension under the symnmf_c name
import symnmf_c as cmod

#Reuse your HW1 K-means helpers (unchanged kmeans.py)
from kmeans import (
    initialize_centroids,
    assign_clusters,
    update_centroids,
    has_converged,
)


def assign_clusters_from_H(H):
    """Convert a SymNMF matrix H (shape n×k) into hard cluster labels"""
    H_np=np.asarray(H, dtype=float)
    if H_np.ndim != 2 or H_np.shape[1] <= 0:
        handle_error()
    return np.argmax(H_np, axis=1) 


def run_kmeans_with_hw1_helpers(points_ll, k, max_iter, epsilon):
    """Run K-means using your HW1 helper functions on in-memory data."""
    n = len(points_ll)
    if not (1 < k < n):
        handle_error()

    dim=len(points_ll[0])
    centroids=initialize_centroids(points_ll, k)

    for _ in range(max_iter):
        cluster_indices=assign_clusters(points_ll, centroids)
        new_centroids=update_centroids(points_ll, cluster_indices, k, dim)
        if has_converged(centroids, new_centroids, epsilon):
            centroids=new_centroids
            break
        centroids=new_centroids

    #Final assignment after convergence
    labels=assign_clusters(points_ll, centroids)
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

        # Load data (list of lists), also keep numpy view for silhouette
        X_ll = load_points(file_name)
        X_np = np.asarray(X_ll, dtype=float) #convert list of lists to np array
        n = X_np.shape[0]
        if not (1 < k < n):
            handle_error()

        # ---- SymNMF path ---- (uses EPS=1e-4, MAX_ITER=300)
        W_ll = cmod.norm(X_ll) #Compute normalized similarity matrix W
        H0 = init_H(W_ll, k) #Randomly initialize H (n×k)

        Hf = cmod.symnmf(H0.tolist(), W_ll, EPS, MAX_ITER, BETA, DEN_EPS)  #Optimize H using SymNMF C extension
        y_nmf = assign_clusters_from_H(Hf) #Convert final H into cluster labels
        nmf_score = float(silhouette_score(X_np, y_nmf, metric="euclidean")) #Evaluate clustering quality

        #KMeans 
        y_km = run_kmeans_with_hw1_helpers(X_ll, k, max_iter=MAX_ITER, epsilon=EPS)
        km_score = float(silhouette_score(X_np, y_km, metric="euclidean"))

        #Output format per spec
        print(f"nmf: {nmf_score:.4f}")
        print(f"kmeans: {km_score:.4f}")

    except Exception:
        handle_error()


if __name__ == "__main__":
    main()
