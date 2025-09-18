import math

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

def printCentroids(centroids):
    """Print centroids as comma-separated values (optional helper)."""
    for C in centroids:
        print(*["%.4f" % x for x in C], sep=",")
