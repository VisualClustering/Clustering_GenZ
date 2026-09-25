import os
import csv
import numpy as np

from PIL import Image, ImageOps, ImageDraw, ImageFont

from Analysis_Helpers import (
    _extract_date_from_filename,
    get_country_from_path
)


def render_random_clusters_panel(
    random_idx_map,
    img_paths,
    labels,
    out_path,
    thumb=180,
    padding=16,
    n_cols=5
):
    """
    Renders 20 random images per cluster.
    Each cluster gets its own block of rows.
    """

    cluster_ids = sorted(random_idx_map.keys())

    max_images = max(len(v) for v in random_idx_map.values())
    rows_per_cluster = int(np.ceil(max_images / n_cols))

    text_height = 55
    header_height = 35
    cell_w = thumb + padding
    cell_h = thumb + text_height + padding

    width = padding + n_cols * cell_w

    cluster_block_height = (
        header_height +
        rows_per_cluster * cell_h +
        padding
    )

    height = padding + len(cluster_ids) * cluster_block_height

    panel = Image.new(
        "RGB",
        (width, height),
        (255, 255, 255)
    )

    draw = ImageDraw.Draw(panel)

    try:
        font = ImageFont.truetype("Arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    for cluster_pos, cl in enumerate(cluster_ids):

        block_y = padding + cluster_pos * cluster_block_height

        cluster_size = np.sum(labels == cl)

        draw.text(
            (padding, block_y),
            f"Cluster {cl:02d} – random sample "
            f"(20 max.; total n={cluster_size})",
            fill=(0, 0, 0),
            font=font
        )

        idxs = random_idx_map[cl]

        for position, i in enumerate(idxs):

            row = position // n_cols
            col = position % n_cols

            x = padding + col * cell_w
            y = block_y + header_height + row * cell_h

            try:
                im = Image.open(img_paths[i])
                im = ImageOps.exif_transpose(im)
                im = im.convert("RGB")

                # Preserve aspect ratio inside thumbnail
                im.thumbnail((thumb, thumb))

                # Centre image in thumbnail area
                image_x = x + (thumb - im.width) // 2
                image_y = y + (thumb - im.height) // 2

                panel.paste(im, (image_x, image_y))

                country_text = get_country_from_path(img_paths[i])
                date_text = _extract_date_from_filename(img_paths[i])

                text = f"{country_text}\n{date_text}"

                bbox = draw.multiline_textbbox(
                    (0, 0),
                    text,
                    font=font,
                    align="center"
                )

                text_w = bbox[2] - bbox[0]

                text_x = x + (thumb - text_w) // 2
                text_y = y + thumb + 5

                draw.multiline_text(
                    (text_x, text_y),
                    text,
                    fill=(0, 0, 0),
                    font=font,
                    align="center"
                )

            except Exception as e:
                print(
                    f"[WARN] Error rendering "
                    f"{img_paths[i]}: {e}"
                )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    panel.save(out_path)

    print(f"[INFO] Random cluster panel saved: {out_path}")


def render_clusters_panel_10(nearest_idx_map, img_paths, labels, out_path, thumb=200, padding=16):
    cluster_ids = sorted(nearest_idx_map.keys())
    n_cols = 10
    n_rows = len(cluster_ids)
    cell = thumb + padding + 75  # extra space for text
    width = padding + n_cols * cell
    height = 50 + n_rows * cell
    panel = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(panel)

    try:
        font = ImageFont.truetype("Arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    for r, cl in enumerate(cluster_ids):
        idxs = nearest_idx_map[cl]  # sorted by centrality

        # Total size per cluster from 'labels'
        cluster_size = np.sum(labels == cl)

        # Title with cluster ID + total size
        draw.text(
            (padding, 20 + r * cell),
            f"Cluster {cl:02d} (n={cluster_size})",
            fill=(0, 0, 0),
            font=font
        )

        for c, i in enumerate(idxs[:n_cols]):
            try:
                with Image.open(img_paths[i]) as img:
                    im = ImageOps.exif_transpose(img)
                    im = im.convert("RGB")
                    im.thumbnail((thumb, thumb))
                    x = padding + c * cell
                    y = 40 + r * cell

                    # Paste image
                    panel.paste(im, (x, y))

                # Caption centered
                country_text = get_country_from_path(img_paths[i])
                date_text = _extract_date_from_filename(img_paths[i])

                text = f"{country_text}\n{date_text}"
                bbox = draw.multiline_textbbox(
                    (0, 0),
                    text,
                    font=font,
                    align="center"
                )

                text_w = bbox[2] - bbox[0]

                text_x = x + (thumb - text_w) // 2
                text_y = y + thumb + 10

                draw.multiline_text(
                    (text_x, text_y),
                    text,
                    fill=(0, 0, 0),
                    font=font,
                    align="center"
                )

            except Exception as e:
                print(f"[WARN] Error with {img_paths[i]}: {e}")


    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    panel.save(out_path)
    print(f"[INFO] Panel saved: {out_path}")

def render_random_clusters_panel_10(
    img_paths,
    labels,
    out_path,
    thumb=200,
    padding=16,
    random_state=42
):

    # Renders 10 random images per cluster
    rng = np.random.default_rng(random_state)

    cluster_ids = sorted(np.unique(labels))

    n_cols = 10
    n_rows = len(cluster_ids)

    cell = thumb + padding + 75
    width = padding + n_cols * cell
    height = 50 + n_rows * cell

    panel = Image.new(
        "RGB",
        (width, height),
        (255, 255, 255)
    )

    draw = ImageDraw.Draw(panel)

    try:
        font = ImageFont.truetype("Arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    for r, cl in enumerate(cluster_ids):

        # All indices belonging to this cluster
        cluster_idxs = np.where(labels == cl)[0]

        cluster_size = len(cluster_idxs)

        # Randomly sample up to 10 images
        n_sample = min(10, cluster_size)

        random_idxs = rng.choice(
            cluster_idxs,
            size=n_sample,
            replace=False
        )

        # Cluster title
        draw.text(
            (padding, 20 + r * cell),
            f"Cluster {cl:02d} – random sample (n={cluster_size})",
            fill=(0, 0, 0),
            font=font
        )

        for c, i in enumerate(random_idxs):

            try:
                with Image.open(img_paths[i]) as img:
                    im = ImageOps.exif_transpose(img)
                    im = im.convert("RGB")
                    im.thumbnail((thumb, thumb))

                    x = padding + c * cell
                    y = 40 + r * cell

                    # Center image inside thumbnail area
                    image_x = x + (thumb - im.width) // 2
                    image_y = y + (thumb - im.height) // 2

                    panel.paste(
                        im,
                        (image_x, image_y)
                    )

                country_text = get_country_from_path(
                    img_paths[i]
                )

                date_text = _extract_date_from_filename(
                    img_paths[i]
                )

                text = f"{country_text}\n{date_text}"

                bbox = draw.multiline_textbbox(
                    (0, 0),
                    text,
                    font=font,
                    align="center"
                )

                text_w = bbox[2] - bbox[0]

                text_x = x + (thumb - text_w) // 2
                text_y = y + thumb + 10

                draw.multiline_text(
                    (text_x, text_y),
                    text,
                    fill=(0, 0, 0),
                    font=font,
                    align="center"
                )

            except Exception as e:
                print(
                    f"[WARN] Error with "
                    f"{img_paths[i]}: {e}"
                )

    os.makedirs(
        os.path.dirname(out_path),
        exist_ok=True
    )

    panel.save(out_path)

    print(
        f"[INFO] Random 10-image cluster panel saved: "
        f"{out_path}"
    )

def save_cluster_filenames(nearest_idx_map, img_paths, out_csv):
    """
    Saves the filenames of the 10 most representative images per cluster.
    """
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster", "rank", "filename"])
        for cl in sorted(nearest_idx_map.keys()):
            idxs = nearest_idx_map[cl]
            for rank, i in enumerate(idxs[:10], start=1):
                writer.writerow([cl, rank, os.path.basename(img_paths[i])])
    print(f"[INFO] CSV file saved: {out_csv}")


def export_all_filenames_with_clusters(labels, img_paths, out_csv):
    """
    Saves all images, along with their cluster labels, to a CSV file,
    sorted by cluster ID and filename.
    """
    rows = []
    for i, path in enumerate(img_paths):
        rows.append([labels[i], os.path.basename(path), path])

    # Sort by cluster ID, then by filename
    rows.sort(key=lambda x: (x[0], x[1]))

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster", "filename", "full_path"])
        writer.writerows(rows)

    print(f"[INFO] CSV file saved with {len(img_paths)} images: {out_csv}")
