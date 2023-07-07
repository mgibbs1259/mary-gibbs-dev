import logging

import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans


def initalize_centroids(data: np.ndarray, k: int) -> np.ndarray:
    """Initializes the centroids for the K-means clustering algorithm

    Args:
        data (np.ndarray): Input data
        k (int): Number of clusters

    Returns:
        np.ndarray: The initial centroids
    """
    rng = np.random.default_rng()
    initial_centroids = rng.choice(data, k, axis=0)
    return initial_centroids


def assign_points_to_centroids(data: np.ndarray, k: int, centroids: np.ndarray) -> list:
    """Assigns each data point to the nearest centroid

    Args:
        data (np.ndarray): Input data 
        k (int): Number of clusters
        centroids (np.ndarray): Current centroids

    Returns:
        list: The cluster assignments for each data point
    """
    distances = np.empty(shape=(data.shape[0], k))
    for i in range(k):
        distances[:, i] = np.sqrt(np.sum((data - centroids[i]) ** 2, axis=1))
    assigned_centroids = distances.argmin(axis=1)
    return assigned_centroids


def update_centroids(data: np.ndarray, k: int, assigned_centroids: list) -> np.ndarray:
    """Updates each centroid by computing the mean using 
    all of the data points assigned to the centroid
    
    Args:
        data (np.ndarray): Input data 
        k (int): Number of clusters
        assigned_centroids (list): Cluster assignments for each data point

    Returns:
        np.ndarray: The updated centroids
    """
    updated_centroids = np.empty(shape=(k, data.shape[1]))
    for i in range(k):
        updated_centroids[i, :] = data[assigned_centroids == i].mean(axis=0)
    return updated_centroids


def stop_kmeans(
    iterations: int,
    max_interations: int,
    previous_centroids: np.ndarray,
    centroids: np.ndarray,
) -> bool:
    """Checks the stopping criterion for the K-means algorithm

    Args:
        iterations (int): Current number of iterations
        max_iterations (int): Maximum number of iterations
        previous_centroids (np.ndarray): The previous centroids
        centroids (np.ndarray): The current centroids

    Returns:
        bool: True if the algorithm should stop, False otherwise
    """
    if iterations > max_interations:
        logging.info(f"Max iteractions {max_interations} reached")
        return True
    return (previous_centroids == centroids).all()


def perform_kmeans(data: np.ndarray, k: int, max_interations: int) -> np.ndarray:
    """Performs K-means clustering on the input data

    Args:
        data (np.ndarray): Input data
        k (int): Number of clusters
        max_iterations (int): Maximum number of iterations

    Returns:
        np.ndarray: The final centroids
    """
    centroids = initalize_centroids(iris_data.data, k)

    iterations = 0
    previous_centroids = None
    while not stop_kmeans(iterations, max_interations, previous_centroids, centroids):
        iterations += 1
        previous_centroids = centroids
        assigned_centroids = assign_points_to_centroids(data, k, centroids)
        centroids = update_centroids(data, k, assigned_centroids)

    return centroids


if __name__ == "__main__":
    iris_data = load_iris()

    centroids = perform_kmeans(data=iris_data.data, k=3, max_interations=5000)
    print(f"My implementation: {centroids}\n")

    sklearn_kmeans_centroids = KMeans(n_clusters=3, random_state=0).fit(iris_data.data)
    print(f"scikit-learn implementation: {sklearn_kmeans_centroids.cluster_centers_}")
