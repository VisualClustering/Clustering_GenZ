
# Reproducibility / TensorFlow configurationn
import joblib, os

# Deactivate oneDNN
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Imports
from pathlib import Path
import hashlib
from collections import Counter
import csv

import numpy as np
import keras_hub


from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

from PIL import Image, ImageOps, ImageDraw, ImageFont
import pillow_heif
pillow_heif.register_heif_opener()


# TensorFlow / Keras
import tensorflow as tf

Dataset_SEED = 42
Panel_SEED=42

tf.keras.utils.set_random_seed(Dataset_SEED)
tf.config.experimental.enable_op_determinism()

from tensorflow.keras.applications.vgg16 import (
    VGG16,
    preprocess_input as vgg16_preprocess
)

from tensorflow.keras.models import Model
from tensorflow.keras.utils import load_img, img_to_array

# Own modules
from Analysis_Helpers import (
    random_images_per_cluster,
    _extract_date_from_filename,
    get_country_from_path,
    export_cluster_country_composition,
    calculate_country_overrepresentation
)
from Clustering import (evaluate_k_values,
                        auto_knee,
                        nearest_to_kmeans_center)

from Plotting import (plot_clusters_pca, plot_countries_pca)

from Output import (
    render_random_clusters_panel,
    render_clusters_panel_10,
    render_random_clusters_panel_10,
    save_cluster_filenames,
    export_all_filenames_with_clusters
)

# Image folder(s)
sample_name = ("GenZ_Madagascar_Morocco_Nepal_n")

image_folders = [
    Path(r"C:\Users\zime9516\Desktop\DatenGenZ\0_Madagaskar\0_gen_z_madagascar_filter\Bilder"),
    Path(r"C:\Users\zime9516\Desktop\DatenGenZ\0_Marokko\0_system_tbdel_filter\Bilder"),
    Path(r"C:\Users\zime9516\Desktop\DatenGenZ\0_Nepal\0_GenZ Nepal\gen.znepal_filter\Bilder"),
]

valid_extensions = {".jpeg", ".jpg", ".webp", ".png", ".heic", ".heif"}

img_paths = sorted(
    str(file)
    for folder in image_folders
    for file in folder.iterdir()
    if file.is_file() and file.suffix.lower() in valid_extensions
)

if len(img_paths) < 30:
    raise RuntimeError(
        f"Only {len(img_paths)} images were found."
        "At least 30 images are required for clustering."
    )

print(f"[INFO] {len(img_paths)} Image files found.")


# Full or balanced?
use_full = input(
    "Use full dataset? (y/n): "
).strip().lower()

if use_full == "y":
    dataset_mode = "full"

else:
    dataset_mode = "balanced202"

    N_PER_COUNTRY = 202
    rng = np.random.default_rng(Dataset_SEED)

    countries = np.array([
        get_country_from_path(path)
        for path in img_paths
    ])

    selected_indices = []

    for country in sorted(set(countries)):
        idxs = np.where(countries == country)[0]

        if len(idxs) < N_PER_COUNTRY:
            raise ValueError(
                f"{country} has only {len(idxs)} images, "
                f"but {N_PER_COUNTRY} were requested."
            )

        chosen = rng.choice(
            idxs,
            size=N_PER_COUNTRY,
            replace=False
        )

        selected_indices.extend(chosen)

    img_paths = [
        img_paths[i]
        for i in selected_indices
    ]

    print(
        f"[INFO] Balanced sample: "
        f"{N_PER_COUNTRY} images per country, "
        f"{len(img_paths)} total."
    )

# Feature extraction: layer selection
feature_mode = input(
    "Choose feature mode "
    "(vgg_fc1 / vgg_fc2 / resnet18 / resnet50): "
).strip().lower()

if feature_mode == "vgg_fc1":
    print("[INFO] VGG16 fc1 features (4096D)")

    base_model = VGG16(
        weights="imagenet",
        include_top=True
    )

    feature_model = Model(
        inputs=base_model.input,
        outputs=base_model.get_layer("fc1").output
    )

    preprocess_fn = vgg16_preprocess
    img_size = (224, 224)
    feature_dim = 4096
    model_type = "keras"


elif feature_mode == "vgg_fc2":
    print("[INFO] VGG16 fc2 features (4096D)")

    base_model = VGG16(
        weights="imagenet",
        include_top=True
    )

    feature_model = Model(
        inputs=base_model.input,
        outputs=base_model.get_layer("fc2").output
    )

    preprocess_fn = vgg16_preprocess
    img_size = (224, 224)
    feature_dim = 4096
    model_type = "keras"

elif feature_mode == "resnet18":
    print("[INFO] ResNet18 ImageNet backbone features (512D)")

    resnet_backbone = keras_hub.models.ResNetBackbone.from_preset(
        "resnet_18_imagenet"
    )

    resnet_converter = keras_hub.layers.ResNetImageConverter.from_preset(
        "resnet_18_imagenet",
        image_size=(224, 224),
        crop_to_aspect_ratio=False,
        pad_to_aspect_ratio=False
    )

    img_size = (224, 224)
    feature_dim = 512
    model_type = "keras_hub_resnet"

elif feature_mode == "resnet50":
    print("[INFO] ResNet50 ImageNet backbone features (2048D)")

    resnet_backbone = keras_hub.models.ResNetBackbone.from_preset(
        "resnet_50_imagenet"
    )

    resnet_converter = keras_hub.layers.ResNetImageConverter.from_preset(
        "resnet_50_imagenet",
        image_size=(224, 224),
        crop_to_aspect_ratio=False,
        pad_to_aspect_ratio=False
    )

    img_size = (224, 224)
    feature_dim = 2048
    model_type = "keras_hub_resnet"

else:
    raise ValueError(f"Unknown feature_mode: '{feature_mode}'. "
                      f"Possible: vgg_fc1, vgg_fc2, resnet18, resnet50")

# Compute or load features
features, valid_img_paths, errors = [], [], []

for i, imgpath in enumerate(img_paths, 1):
    try:
        stat = os.stat(imgpath)

        # Cache key
        # Image path + file size + modification time + CNN used
        cache_key = (
            f"{os.path.abspath(imgpath)}|"
            f"{stat.st_size}|"
            f"{stat.st_mtime_ns}|"
            f"{feature_mode}"
        )

        cache_hash = hashlib.sha1(cache_key.encode("utf-8")).hexdigest()[:16]

        stem = os.path.splitext(os.path.basename(imgpath))[0]

        # Feature folder
        feature_folder = Path(imgpath).parent / "_features"
        feature_folder.mkdir(parents=True, exist_ok=True)

        # Save features
        savepath = feature_folder / f"{stem}_{feature_mode}_{cache_hash}.pkl"

        # Load features
        loaded_from_cache = savepath.exists()

        if loaded_from_cache:
            vec = joblib.load(savepath)

        else:
            img = load_img(
                imgpath,
                target_size=img_size,
                color_mode="rgb",
                keep_aspect_ratio=False
            )

            arr = img_to_array(img)[None, ...]

            if model_type == "keras":
                # VGG16:
                # Direct resize ("squishing") to 224 × 224.
                # The complete image is retained, but the aspect ratio
                # is not preserved.
                arr = preprocess_fn(arr)

                vec = feature_model.predict(
                    arr,
                    verbose=0
                ).astype(np.float32)

            elif model_type == "keras_hub_resnet":
                # Image has already been directly resized to 224 × 224
                # by load_img (squishing).
                arr = resnet_converter(arr)

                feature_map = resnet_backbone.predict(
                    arr,
                    verbose=0
                ).astype(np.float32)

                vec = feature_map.mean(axis=(1, 2))

        # Reduce to an one-dimensional feature vector
        vec = np.asarray(vec,
            dtype=np.float32).reshape(-1)

        # Check whether the dimension matches the selected CNN
        if len(vec) != feature_dim:
            raise ValueError(
                f"An unexpected aspect of the feature:"
                f"{len(vec)} statt {feature_dim}"
            )
        # Save the recalculated feature
        if not loaded_from_cache:
            joblib.dump(vec, savepath)

        features.append(vec)
        valid_img_paths.append(imgpath)

    except Exception as e:
        errors.append((imgpath, str(e)))
        print(f"[WARN] Skipped {imgpath} — {e}")

    if i % 50 == 0 or i == len(img_paths):
        print(f"[EXTRACT] {i}/{len(img_paths)} processed …")

img_paths = valid_img_paths

if not features:
    raise RuntimeError("No image features could be generated.")

X = np.vstack(features)

print("[INFO] Feature matrix:", X.shape)
print(f"[INFO] Skipped files: {len(errors)}")


# Normalize features and PCA
X = normalize(X, norm="l2")

max_components = min(400, X.shape[0] - 1, X.shape[1])

if max_components < 2:
    raise RuntimeError(
        "There are too few images for a meaningful PCA and cluster analysis."
    )

pca = PCA(
    n_components=max_components,
    random_state=Dataset_SEED,
)

X_reduced = pca.fit_transform(X)

print(
    f"[INFO] PCA: {X.shape[1]} → {X_reduced.shape[1]} Dimensions; "
    f"explained variance: "
    f"{pca.explained_variance_ratio_.sum():.2%}"
)


# Elbow + multi-criteria evaluation of k
n_samples = len(X_reduced)
max_k = min(20, n_samples - 1)

if max_k < 2:
    raise RuntimeError(
        "Not enough images for the cluster count analysis."
    )

K = list(range(1, max_k + 1))
k_values = list(range(2, max_k + 1))

# Elbow
inertias = []

for k in K:
    km = KMeans(
        n_clusters=k,
        random_state=Dataset_SEED,
        n_init=30
    )
    km.fit(X_reduced)
    inertias.append(km.inertia_)

# Multi-criteria evaluation
k_metrics = evaluate_k_values(
    X_reduced,
    k_values=k_values
)

# 3b) Auxiliary function: Auto-Elbow
k_elbow_estimated = auto_knee(K, inertias)

k_silhouette = max(
    k_metrics,
    key=lambda row: row["silhouette"]
)["k"]

k_calinski = max(
    k_metrics,
    key=lambda row: row["calinski_harabasz"]
)["k"]

k_davies = min(
    k_metrics,
    key=lambda row: row["davies_bouldin"]
)["k"]


print("\n[INFO] Suggestions for k:")
print(f"  Elbow estimated:    k={k_elbow_estimated}")
print(f"  Silhouette:         k={k_silhouette}")
print(f"  Calinski-Harabasz:  k={k_calinski}")
print(f"  Davies-Bouldin:     k={k_davies}")


# User-defined cluster number
while True:
    try:
        k_best = int(
            input(
                f"How many clusters? "
                f"(2–{max_k}; Silhouette={k_silhouette}, Elbow={k_elbow_estimated}): "
            ).strip()
        )

        if 2 <= k_best <= max_k:
            break

        print(f"Please enter a number between 2 and {max_k}.")

    except ValueError:
        print("Invalid input. Please enter a whole number.")

# Preview output folder
preview_dir = f"cluster_previews_{sample_name}_{feature_mode}_{dataset_mode}_k{k_best:02d}_pca400"
os.makedirs(preview_dir, exist_ok=True)

# Save the CSV file for future reference
metrics_file = Path(preview_dir) / f"k_selection_metrics_{feature_mode}.csv"

with open(metrics_file, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "k",
            "silhouette",
            "calinski_harabasz",
            "davies_bouldin",
        ]
    )
    writer.writeheader()
    writer.writerows(k_metrics)

print(f"[INFO] Metrics saved: {metrics_file}")

# ave Elbow plot
plt.figure(figsize=(8, 5))
plt.plot(K, inertias, marker="o")
plt.xlabel("Number of clusters k")
plt.ylabel("Within-cluster sum of squares (Inertia)")
plt.title("Elbow Method")
plt.xticks(K)
plt.grid(alpha=0.3)

plt.savefig(
    os.path.join(preview_dir, "elbow_plot.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# Final clustering
km_final = KMeans(
    n_clusters=k_best,
    random_state=Dataset_SEED,
    n_init=30
)
labels = km_final.fit_predict(X_reduced)
counts = Counter(labels)
print("[INFO] Cluster sizes:", dict(counts))


# Random sample: 20 images per cluster
random_idx_map = random_images_per_cluster(
    labels,
    n_per_cluster=20,
    random_state=Panel_SEED
)

render_random_clusters_panel(
    random_idx_map,
    img_paths,
    labels,
    out_path=os.path.join(
        preview_dir,
        f"clusters_random20_panel_{sample_name}_{feature_mode}_{dataset_mode}_k{k_best:02d}_pca400.png"
    )
)

# Generate panels
nearest = 10

nearest_idx_map_kmeans = nearest_to_kmeans_center(X_reduced, labels, km_final, top_n=nearest)

render_clusters_panel_10(nearest_idx_map_kmeans,
                         img_paths,
                         labels,
                         out_path=os.path.join(
                             preview_dir,
                             f"clusters_kmeans10_panel_{sample_name}_{feature_mode}_{dataset_mode}_k{k_best:02d}_pca400.png"
                         )
)

render_random_clusters_panel_10(
    img_paths,
    labels,
    out_path=os.path.join(
        preview_dir,
        f"clusters_kmeans10_random_panel_{sample_name}_{feature_mode}_{dataset_mode}_k{k_best:02d}_pca400.png"
    ),
    random_state=Panel_SEED
)

# KMeans variant
save_cluster_filenames(
    nearest_idx_map_kmeans,
    img_paths,
    os.path.join(preview_dir, "cluster_kmeans10_filenames.csv")
)

# Graphs_PCA
plot_clusters_pca(
    X_reduced,
    labels,
    kmeans_model=km_final,
    output_path=os.path.join(preview_dir, "clusters_3d_pca.png")
)

plot_countries_pca(
    X_reduced,
    img_paths,
    labels,
    output_path=os.path.join(
        preview_dir,
        "countries_clusters_3d_pca.png"
    )
)
# Export CSV with all images
export_all_filenames_with_clusters(
    labels,
    img_paths,
    os.path.join(preview_dir, "all_images_with_clusters.csv"))

(
    df_cluster_country,
    country_counts,
    country_percent_cluster,
    country_percent_country
) = export_cluster_country_composition(
    labels,
    img_paths,
    preview_dir
)

overrep_df = calculate_country_overrepresentation(
    img_paths,
    labels
)

overrep_df.to_csv(
    os.path.join(
        preview_dir,
        "cluster_country_overrepresentation.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)
# Save model and metadata
run_metadata = {
    "feature_mode": feature_mode,
    "k": k_best,
    "image_paths": img_paths,
    "labels": labels,
    "pca": pca,
    "pca_n_components": pca.n_components_,
    "pca_explained_variance": pca.explained_variance_ratio_.sum() * 100,
    "kmeans": km_final
}

joblib.dump(
    run_metadata,
    os.path.join(preview_dir, "clustering_model_and_metadata.pkl")
)

print("[INFO] Model, PCA and metadata saved.")
