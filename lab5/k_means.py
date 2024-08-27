import numpy as np

def initialize_centroids_forgy(data, k):
    # TODO implement random initialization
    centroids_indices = np.random.choice(len(data), k, replace=False)
    centroids = data[centroids_indices]
    return centroids


def initialize_centroids_kmeans_pp(data, k):
    # TODO implement kmeans++ initizalization
    k_samples = data.shape[0]
    centroid_indices = [np.random.randint(0, k_samples)]
    max_distances = np.zeros(k_samples)
    for _ in range(1, k):
        distances = np.linalg.norm(data - data[centroid_indices[-1]], axis=1)
        max_distances = np.maximum(max_distances, distances)
        farthest_idx = np.argmax(max_distances)
        centroid_indices.append(farthest_idx)
    centroids = data[centroid_indices]
    return centroids


def assign_to_cluster(data, centroid):
    # TODO find the closest cluster for each data point
    distances = np.sqrt(((data - centroid[:, np.newaxis]) ** 2).sum(axis=2))
    return np.argmin(distances, axis=0)

def update_centroids(data, assignments):
    # TODO find new centroids based on the assignments
    unique_values = np.unique(assignments)
    k = len(unique_values)
    centroids = np.zeros((k, data.shape[1]))
    for i in range(k):
        centroids[i] = np.mean(data[assignments == i], axis=0)
    return centroids

def mean_intra_distance(data, assignments, centroids):
    return np.sqrt(np.sum((data - centroids[assignments, :])**2))

def k_means(data, num_centroids, kmeansplusplus= False):
    # centroids initizalization
    if kmeansplusplus:
        centroids = initialize_centroids_kmeans_pp(data, num_centroids)
    else: 
        centroids = initialize_centroids_forgy(data, num_centroids)

    assignments = assign_to_cluster(data, centroids)
    for i in range(100): # max number of iteration = 100
        print(f"Intra distance after {i} iterations: {mean_intra_distance(data, assignments, centroids)}")
        centroids = update_centroids(data, assignments)
        new_assignments = assign_to_cluster(data, centroids)
        if np.all(new_assignments == assignments): # stop if nothing changed
            break
        else:
            assignments = new_assignments

    return new_assignments, centroids, mean_intra_distance(data, new_assignments, centroids)         

