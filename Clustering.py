import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)

def evaluate_k_values(
    X,
    k_values,
    random_state=42
):
    """
    For each k, check:
    - Silhouette: the higher, the better
    - Calinski-Harabasz: the higher, the better
    - Davies-Bouldin: the lower, the better
    """

    n_samples = len(X)
    results = []

    for k in k_values:
        km_full = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=30
        )

        labels_full = km_full.fit_predict(X)

        silhouette = silhouette_score(
            X,
            labels_full,
            sample_size=min(5000, n_samples),
            random_state=random_state
        )

        calinski = calinski_harabasz_score(X, labels_full)
        davies = davies_bouldin_score(X, labels_full)

        result = {
            "k": k,
            "silhouette": float(silhouette),
            "calinski_harabasz": float(calinski),
            "davies_bouldin": float(davies),
        }

        results.append(result)

        print(
            f"[K={k:02d}] "
            f"Silhouette={silhouette:.3f} | "
            f"CH={calinski:.1f} | "
            f"DB={davies:.3f} | "
        )

    return results


def auto_knee(x, y):
    """
    x: list/array of k-values
    y: list/array of metrics
    Method: Maximum Distance to Line
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x_n = (x - x.min()) / (x.max() - x.min() + 1e-12)
    y_n = (y - y.min()) / (y.max() - y.min() + 1e-12)

    p1 = np.array([x_n[0], y_n[0]])
    p2 = np.array([x_n[-1], y_n[-1]])

    v = p2 - p1
    v /= (np.linalg.norm(v) + 1e-12)

    dists = []

    for xi, yi in zip(x_n, y_n):
        p = np.array([xi, yi])
        proj_len = np.dot(p - p1, v)
        proj = p1 + proj_len * v
        d = np.linalg.norm(p - proj)
        dists.append(d)

    idx = int(np.argmax(dists))

    return int(x[idx])

def nearest_to_kmeans_center(
    X,
    labels,
    kmeans_model,
    top_n=10
):
    results = {}

    for cl in sorted(set(labels)):
        idxs = np.where(labels == cl)[0]
        center = kmeans_model.cluster_centers_[cl]
        dists = np.linalg.norm(X[idxs] - center, axis=1)
        nearest = idxs[np.argsort(dists)[:top_n]]
        results[cl] = list(nearest)

    return results
