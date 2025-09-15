import math
import sys #newly added

def load_points():
    """
    only use this function in the first iteration.
    Reads data points from a file.

    Parameters:
        input_data (str): Path to the input text file.

    Returns:
        list[list[float]]: A list of points, where each point is a list of floats.
    """
    points = []  
    try:  #new added
        for line in sys.stdin:  #new added
            arr = list(map(float, line.strip().split(",")))  
            points.append(arr)  
        return points  
    except:  #new added
        print("An Error Has Occurred")  #new added
        sys.exit(1)  #new added

def initialize_centroids(points, K):
    """
    only use this function in the first iteration.
    Selects the first K points from the dataset to serve as the initial centroids.

    Parameters:
        points (list[list[float]]): The list of all data points.
        K (int): The number of clusters.

    Returns:
        list[list[float]]: A list of initial centroids. #returns list of list
    """
    return [points[i][:] for i in range(K)]  # copy of first K points

def assign_clusters(points, centroids):
    """
    Assigns each point to the index of the closest centroid.

    Parameters:
        points (list[list[float]]): The dataset points.
        centroids (list[list[float]]): The current list of centroids.

    Returns:
        list[int]: A list where the i-th value is the index of the centroid assigned to point i
        - simply to know what centroid is closest to point p , -> cluster_indices[points.index(p)]=x , that x is the index of the closest centroid [1...K]-
    """
    cluster_indices = []
    for p in points:
        min_dist = float('inf')
        min_index = -1
        #enumerate(centroids) gives both:, j: the index (0, 1, 2, ...) , c: the value (i.e., each centroid)
        for j, c in enumerate(centroids): 
            d = dist(p, c) #both point and centroids are arrays each array is of the same size  
            if d < min_dist:
                min_dist = d
                min_index = j
        cluster_indices.append(min_index) #the index of the closest centroid to point p is found and appended to the cluster_indeces list, that len(cluster_indices)=len(points) 
    return cluster_indices

def update_centroids(points, cluster_indices, K, dim):
    """
    Calculates new centroids as the mean of all assigned points in each cluster.

    Parameters:
        points (list[list[float]]): The list of points.
        cluster_indices (list[int]): Cluster assignment for each point.
        K (int): Number of clusters.
        dim (int): Dimensionality of each point.

    Returns:
        list[list[float]]: Updated centroid coordinates.
    """
    new_centroids = [[0.0] * dim for _ in range(K)] # creating an empty (0.0) list of lists that each inner list of size dim
    counts = [0] * K #counts the number of points for each centroid
    # i is the index in cluster_indices and also the index of this point in points list
    # cluster is the index of the appropriate centroid
    for i, cluster in enumerate(cluster_indices):
        for d in range(dim): #iterating over the elements of the point
            new_centroids[cluster][d] += points[i][d] #adding the elements of point i to the list of the matches centroid in the new centroids list
        counts[cluster] += 1 

    for j in range(K):
        if counts[j] == 0:#if one centroid is empty of suitable points do NoT do anything .. just continue
            continue  # Avoid division by zero
        for d in range(dim):
            new_centroids[j][d] /= counts[j]# else calc the mean for each dimintion of the new centroid

    return new_centroids

def has_converged(old_centroids, new_centroids, epsilon):
    """
    Checks if the centroids have moved less than epsilon in all dimensions.

    Parameters:
        old_centroids (list[list[float]]): Previous iteration centroids.
        new_centroids (list[list[float]]): Current iteration centroids.
        epsilon (float): Convergence threshold.

    Returns:
        bool: True if all centroids moved less than epsilon.
    """
    for c_old, c_new in zip(old_centroids, new_centroids):
        if dist(c_old, c_new) >= epsilon:
            return False
    return True #only if all true return true 

def dist(p, q):
    """
    Computes the Euclidean distance between two points.

    Parameters:
        p (list[float]): First point.
        q (list[float]): Second point.

    Returns:
        float: The Euclidean distance between p and q.
    """
    return math.sqrt(sum((pi - qi) ** 2 for pi, qi in zip(p, q)))

def printCentroids(Centroids):
    """
    Prints each centroid as a comma-separated line.

    Parameters:
        Centroids (list[list[float]]): Final list of centroids.
    """
    for C in Centroids:
        print(*["%.4f" % x for x in C], sep=",")  #new added (added %.4f formatting here)

def k_means(K, iter=400):  #new added
    """
    Main K-means clustering algorithm.

    Parameters:
        K (int): Number of clusters.
        iter (int): Maximum number of iterations to perform.
    """
    epsilon = 0.001  # Convergence threshold
    points = load_points()  #new added

    if not (1 < K < len(points)) :  #new added
        print("Incorrect number of clusters!")  #new added
        sys.exit(1)  #new added
 
    if not (1 < iter < 1000 ) :  #new added
        print("Incorrect maximum iteration!")  #new added
        sys.exit(1)  #new added

    dim = len(points[0])  # Dimensionality of each point
    centroids = initialize_centroids(points, K)

    for _ in range(iter): #the underscore means that we aint going to use the loops indx
        cluster_indices = assign_clusters(points, centroids)
        new_centroids = update_centroids(points, cluster_indices, K, dim)
        if has_converged(centroids, new_centroids, epsilon): #if all differences are < epsilon just break and print
            break
        centroids = new_centroids

    printCentroids(centroids)
    return 0  

if __name__ == "__main__":  # is this write
    if len(sys.argv) not in [2, 3]:
        print("Invalid number of arguments!")
        sys.exit(1)

    arg = sys.argv[1]  # new line
    try:
        val = float(arg)  # new line
        if val.is_integer():
            K = int(val)  # new line
        else:
            print("Incorrect number of clusters!")  # new line
            sys.exit(1) 
    except ValueError:
        print("Incorrect number of clusters!")
        sys.exit(1)

    if len(sys.argv) == 3:  # new line
        iter_arg = sys.argv[2]  # new line
        try:
            iter_val = float(iter_arg)  # new line
            if iter_val.is_integer():
                iter = int(iter_val)  # new line
            else:
                print("Incorrect maximum iteration!")  # new line
                sys.exit(1)  # new line
        except ValueError:
            print("Incorrect maximum iteration!")
            sys.exit(1)
    else:
        iter = 400  # default iter = 400

    k_means(K, iter)


 
        