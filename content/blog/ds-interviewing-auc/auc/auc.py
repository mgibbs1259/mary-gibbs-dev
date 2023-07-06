import logging

import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans


def initalize_centroids(data: np.ndarray, k: int) -> np.ndarray:
    rng = np.random.default_rng()
    initial_centroids = rng.choice(data, k, axis=0)
    return initial_centroids


def assign_points_to_centroids(data: np.ndarray, k: int, centroids: np.ndarray) -> list:
    distances = np.empty(shape=(data.shape[0], k))
    for i in range(k):
        distances[:, i] = np.sqrt(np.sum((data - centroids[i]) ** 2, axis=1))
    assigned_centroids = distances.argmin(axis=1)
    return assigned_centroids


def update_centroids(data: np.ndarray, k: int, assigned_centroids: list) -> np.ndarray:
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
    if iterations > max_interations:
        logging.info(f"Max iteractions {max_interations} reached")
        return True
    return (previous_centroids == centroids).all()


def perform_kmeans(data: np.ndarray, k: int, max_interations: int) -> np.ndarray:
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
