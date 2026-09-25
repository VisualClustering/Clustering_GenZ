import os
import re
import numpy as np
import pandas as pd

from PIL import Image, ImageOps, ImageDraw, ImageFont


# Random images
def random_images_per_cluster(labels, n_per_cluster=20, random_state=42):
    labels = np.asarray(labels)
    rng = np.random.default_rng(random_state)

    results = {}

    for cl in sorted(set(labels)):
        idxs = np.where(labels == cl)[0]

        n_select = min(n_per_cluster, len(idxs))

        selected = rng.choice(
            idxs,
            size=n_select,
            replace=False
        )

        results[cl] = list(selected)

    return results

def _extract_date_from_filename(fname):

    m = re.search(r'(20\d{2})[-_\.](\d{2})[-_\.](\d{2})', fname)

    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    return os.path.basename(fname)


def get_country_from_path(img_path):
    path_lower = str(img_path).lower()

    if "madagaskar" in path_lower or "madagascar" in path_lower:
        return "Madagascar"
    elif "marokko" in path_lower or "morocco" in path_lower:
        return "Morocco"
    elif "nepal" in path_lower:
        return "Nepal"
    else:
        return "Unknown"


# Clusters x country
def export_cluster_country_composition(
    labels,
    img_paths,
    output_dir
):
    """
    Creates three country × cluster tables:

    1. Absolute counts
    2. Percent composition within each cluster
    3. Percent distribution within each country
    """

    countries = [
        get_country_from_path(path)
        for path in img_paths
    ]

    df = pd.DataFrame({
        "cluster": labels,
        "country": countries,
        "image_path": img_paths
    })
  
    # 1. Absolute numbers

    counts = pd.crosstab(
        df["cluster"],
        df["country"]
    )

    counts["Total"] = counts.sum(axis=1)

    counts.to_csv(
        os.path.join(
            output_dir,
            "cluster_country_counts.csv"
        ),
        encoding="utf-8-sig"
    )

    """
    2. Percentage WITHIN each cluster
    
    Question:
    "Of the images in Cluster 4, what percentage comes from each country?"
    """
    
    cluster_percent = pd.crosstab(
        df["cluster"],
        df["country"],
        normalize="index"
    ) * 100

    cluster_percent.to_csv(
        os.path.join(
            output_dir,
            "cluster_country_percent_within_cluster.csv"
        ),
        encoding="utf-8-sig"
    )

    """
    3. Percentage WITHIN each country
    
     Question:
     "Of all Moroccan images, what percentage falls into Cluster 4?"
    """

    country_percent = pd.crosstab(
        df["cluster"],
        df["country"],
        normalize="columns"
    ) * 100

    country_percent.to_csv(
        os.path.join(
            output_dir,
            "cluster_country_percent_within_country.csv"
        ),
        encoding="utf-8-sig"
    )

    # Print results

    print("\n[INFO] Country composition – counts:")
    print(counts)

    print("\n[INFO] Country composition – % within cluster:")
    print(cluster_percent.round(1))

    print("\n[INFO] Cluster distribution – % within country:")
    print(country_percent.round(1))

    return (
        df,
        counts,
        cluster_percent,
        country_percent
    )

def calculate_country_overrepresentation(img_paths, labels):

    labels = np.asarray(labels)

    countries = np.array([
        get_country_from_path(path)
        for path in img_paths
    ])

    unique_countries = sorted(set(countries))
    unique_clusters = sorted(set(labels))

    # Overall proportion of each country
    overall = {
        country: np.mean(countries == country)
        for country in unique_countries
    }

    rows = []

    for cl in unique_clusters:

        cluster_mask = labels == cl
        n_cluster = cluster_mask.sum()

        for country in unique_countries:

            observed_share = np.mean(
                countries[cluster_mask] == country
            )

            expected_share = overall[country]

            ratio = (
                observed_share / expected_share
                if expected_share > 0
                else np.nan
            )

            rows.append({
                "cluster": cl,
                "country": country,
                "n_cluster": n_cluster,
                "observed_percent":
                    observed_share * 100,
                "overall_percent":
                    expected_share * 100,
                "representation_ratio":
                    ratio
            })

    return pd.DataFrame(rows)
