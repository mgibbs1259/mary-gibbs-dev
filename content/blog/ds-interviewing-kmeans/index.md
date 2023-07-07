---
title: "Data Science Interviewing Pt. 2: KMeans Edition"
date: "2022-03-12"
description: "Data science interviewing is interesting. In this several part series, I will explain/implement solutions to problems that I have come across during the data science interview process. These posts serve as practice, but I hope that others will find them useful as well. This time it's KMeans edition."
---

Data science is a broad field, which results in an extremely varied interview process. It's like the Wild West; there are no rules, no structure, nothing... During a past interview, I was subjected to an hour and a half coding challenge in which I was asked to implement KMeans from scratch in my programing language of choice. Giddy up cowperson, in this post, I'll provide an overview of KMeans followed by my basic implementation of it in Python.

![](wildwest.jpeg)

## What is KMeans?

KMeans is an unsupervised clustering algorithm.

It assumes that...

-    You have some data
-    You don't have any known labels for the data (unsupervised)
-    You want to find some structure in the data based on the properties of the data, so that similar observations can be grouped or "clustered" together (clustering)

KMeans divides the data into a predetermined number (_k_) of clusters with a center point known as the centroid. The KMeans algorithm minimizes the sum of squares distance between every obervation and its centroid, maximizing the sum of squares distance between clusters.

KMeans has applications in customer segmentation, document classification, basic recommendation systems, etc. As a side note, I used KMeans in my personal projet SpotiHue to find the most prominent colors in the album artwork of the currently playing song on my Spotify account. I'll discuss this project in more detail in a future blog post.

## Basic Implementation of KMeans in Python

Let's assume an _n_ x _m_ matrix and _k_ clusters.

1. Randomly initialize centroids

```
def initalize_centroids(data: np.ndarray, k: int) -> np.ndarray:
    rng = np.random.default_rng()
    initial_centroids = rng.choice(data, k, axis=0)
    return initial_centroids
```

2. Compute distance between each observation and each centroid; assign each observation to the closest centroid

```
def assign_points_to_centroids(data: np.ndarray, k: int, centroids: np.ndarray) -> list:
    distances = np.empty(shape=(data.shape[0], k))
    for i in range(k):
        distances[:, i] = np.sqrt(np.sum((data-centroids[i])**2, axis=1))
    assigned_centroids = distances.argmin(axis=1)
    return assigned_centroids
```

3. Update each centroid by computing the mean using all of the data points assigned to the centroid

```
def update_centroids(data: np.ndarray, k: int, assigned_centroids: list) -> np.ndarray:
    updated_centroids = np.empty(shape=(k, data.shape[1]))
    for i in range(k):
        updated_centroids[i, :] = (data[assigned_centroids == i].mean(axis=0))
    return updated_centroids
```

4. Repeat 2. and 3. until a stopping condition is met or until convergence

```
def stop_kmeans(
        iterations: int,
        max_interations: int,
        previous_centroids: np.ndarray,
        centroids: np.ndarray
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
```

I decided to test my implementation using the infamous [Iris Dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html). I compared my implementation to scikit-learn's implementation.

```
My implementation: [[5.006      3.428      1.462      0.246     ]
 [5.9016129  2.7483871  4.39354839 1.43387097]
 [6.85       3.07368421 5.74210526 2.07105263]]

scikit-learn's implementation: [[5.006      3.428      1.462      0.246     ]
 [5.9016129  2.7483871  4.39354839 1.43387097]
 [6.85       3.07368421 5.74210526 2.07105263]]
```

Success!

## Full Code

```
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
```
