from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from collections import Counter
from Analysis_Helpers import get_country_from_path


def plot_clusters_pca(
    X,
    labels,
    kmeans_model=None,
    title="Clusters in 3D PCA space",
    random_state=42,
    output_path="cluster_previews/clusters_3d_pca.png",
    dpi=300
):
    # Creates and saves a 3D PCA visualisation of the clusters.
    if X.shape[0] < 3 or X.shape[1] < 3:
        raise ValueError(
            "For 3D PCA, there must be at least three images"
            "and at least three feature dimensions."
        )

    reducer = PCA(n_components=3, random_state=random_state)
    X_3d = reducer.fit_transform(X)

    counts = Counter(labels)

    colors = [
        "#1f77b4", "#2ca02c", "#ff7f0e", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf"
    ]

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    for cl in sorted(set(labels)):
        idxs = np.where(labels == cl)[0]

        ax.scatter(
            X_3d[idxs, 0],
            X_3d[idxs, 1],
            X_3d[idxs, 2],
            c=colors[cl % len(colors)],
            s=30,
            alpha=0.7,
            label=f"Cluster {cl} (n={counts[cl]})"
        )

    if kmeans_model is not None:
        centers_3d = reducer.transform(kmeans_model.cluster_centers_)

        ax.scatter(
            centers_3d[:, 0],
            centers_3d[:, 1],
            centers_3d[:, 2],
            c="black",
            s=180,
            alpha=0.9,
            marker="X",
            label="Cluster centers"
        )

    ax.set_title(title)
    ax.set_xlabel("First principal component")
    ax.set_ylabel("Second principal component")
    ax.set_zlabel("Third principal component")
    ax.legend(loc="best")

    plt.tight_layout()

    # Create a folder if necessary and save the image
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        output_file,
        dpi=dpi,
        bbox_inches="tight"
    )

    print(f"[INFO] 3D PCA graph saved: {output_file}")

    plt.show()
    plt.close(fig)

def plot_countries_pca(
    X,
    img_paths,
    labels,
    title="Datasets and clusters in 3D PCA space",
    random_state=42,
    output_path="cluster_previews/countries_3d_pca.png",
    dpi=300
):
    from pathlib import Path
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    import numpy as np

    countries = np.array([
        get_country_from_path(path)
        for path in img_paths
    ])

    labels = np.asarray(labels)

    reducer = PCA(
        n_components=3,
        random_state=random_state
    )

    X_3d = reducer.fit_transform(X)

    country_colors = {
        "Madagascar": "#F4D03F",  # goldgelb
        "Morocco": "#48C9B0",  # mint/türkis
        "Nepal": "#AF7AC5"  # helles violett
    }

    cluster_colors = [
        "#1f77b4",
        "#2ca02c",
        "#ff7f0e",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf"
    ]

    fig = plt.figure(figsize=(11, 9))

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    # Countries = point colors
    for country in sorted(set(countries)):

        idxs = np.where(
            countries == country
        )[0]

        color = country_colors.get(
            country,
            "gray"
        )

        ax.scatter(
            X_3d[idxs, 0],
            X_3d[idxs, 1],
            X_3d[idxs, 2],
            c=color,
            s=28,
            alpha=0.95,
            label=f"{country} (n={len(idxs)})"
        )

    # Clusters = transparent wireframe spheres
    for cl in sorted(set(labels)):

        idxs = np.where(
            labels == cl
        )[0]

        cluster_points = X_3d[idxs]

        center = cluster_points.mean(
            axis=0
        )

        distances = np.linalg.norm(
            cluster_points - center,
            axis=1
        )

        radius = np.percentile(
            distances,
            75
        )

        u = np.linspace(
            0,
            2 * np.pi,
            30
        )

        v = np.linspace(
            0,
            np.pi,
            20
        )

        x = (
            center[0]
            + radius
            * np.outer(
                np.cos(u),
                np.sin(v)
            )
        )

        y = (
            center[1]
            + radius
            * np.outer(
                np.sin(u),
                np.sin(v)
            )
        )

        z = (
            center[2]
            + radius
            * np.outer(
                np.ones_like(u),
                np.cos(v)
            )
        )

        color = cluster_colors[
            cl % len(cluster_colors)
        ]

        ax.plot_wireframe(
            x,
            y,
            z,
            color=color,
            alpha=0.18,
            linewidth=0.7
        )

        ax.text(
            center[0],
            center[1],
            center[2],
            f"C{cl}",
            color="black",
            fontsize=10
        )


    # Labels
    ax.set_title(title)

    ax.set_xlabel(
        "First principal component"
    )

    ax.set_ylabel(
        "Second principal component"
    )

    ax.set_zlabel(
        "Third principal component"
    )

    ax.legend(
        title="Dataset",
        loc="best"
    )

    plt.tight_layout()


    # Save
    output_file = Path(
        output_path
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fig.savefig(
        output_file,
        dpi=dpi,
        bbox_inches="tight"
    )

    print(
        f"[INFO] 3D country/cluster PCA saved: "
        f"{output_file}"
    )

    plt.show()
    plt.close(fig)
