import os
import math
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import uniform_filter, distance_transform_edt, label
from skimage.morphology import (
    binary_opening, binary_closing, binary_dilation,
    remove_small_objects, remove_small_holes, disk
)
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from shapely import wkt

# --- Parameters: define input/output paths and algorithm settings ---
input_folder = '/Volumes/External/TJ/030_autoSARoutput_040626/ContrastRatio'
output_folder = '/Volumes/External/TJ/040_segoutput'
wkt_aoi = 'POLYGON ((-117.268066 32.407792, -117.071686 32.407792, -117.071686 32.716822, -117.268066 32.716822, -117.268066 32.407792))'
threshold_value = .35
outfall_shapefile = '/Volumes/External/TJ/015_shapefiles/outflow_PB_TJ/Outflow.shp'
decay_scale = 4000.0
cluster_prob_threshold = 0.4

# Turn debug JPG export on/off
DEBUG = True


def save_debug_jpg(arr, out_path, title='', cmap='viridis', vmin=None, vmax=None):
    """
    Save a NumPy array as a JPG for visual debugging.
    Works for continuous rasters and binary masks.
    """

    arr = np.asarray(arr)

    plt.figure(figsize=(8, 8))

    if arr.dtype == bool:
        plt.imshow(arr, cmap='gray', interpolation='nearest')
    else:
        plt.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
        plt.colorbar(shrink=0.75)

    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()


def process_image(image_path, mask_path, prob_path,
                  wkt_aoi, threshold_value,
                  outfall_shapefile, decay_scale,
                  cluster_prob_threshold,
                  debug=False,
                  debug_folder=None):

    # Create a base name for debug JPGs
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    if debug:
        if debug_folder is None:
            debug_folder = os.path.join(os.path.dirname(mask_path), 'debug_jpgs')
        os.makedirs(debug_folder, exist_ok=True)

    # 1. Open the source GeoTIFF to get data, metadata, and geotransform
    with rasterio.open(image_path) as src:
        img = src.read(1).astype(np.float32)
        nodata = src.nodata
        profile = src.profile
        transform = src.transform
        crs = src.crs
        h, w = src.height, src.width

    if debug:
        save_debug_jpg(
            img,
            os.path.join(debug_folder, f'{base_name}_step01_original.jpg'),
            title='Step 1: Original Image',
            cmap='gray'
        )

    # 2. Low-pass filter with valid-data weighting
    valid = (~np.isnan(img)).astype(np.float32)
    filled = np.nan_to_num(img, nan=0.0)

    num = uniform_filter(filled, size=3, mode='nearest')
    den = uniform_filter(valid, size=3, mode='nearest')

    smooth = np.full_like(img, np.nan, dtype=np.float32)
    mask_den = den > 0
    smooth[mask_den] = num[mask_den] / den[mask_den]

    if debug:
        save_debug_jpg(
            smooth,
            os.path.join(debug_folder, f'{base_name}_step02_smooth.jpg'),
            title='Step 2: Smoothed Image',
            cmap='gray'
        )

    # 3. Threshold & morphological cleanup
    # 3. Valid-data mask
    if nodata is None:
        signal = ~np.isnan(smooth)
    else:
        signal = (~np.isnan(smooth)) & (img != nodata)

    mask = signal

    if debug:
        save_debug_jpg(
            signal,
            os.path.join(debug_folder, f'{base_name}_step03_signal_threshold.jpg'),
            title=f'Step 3a: Thresholded Signal > {threshold_value}',
            cmap='gray'
        )

    opened = binary_opening(signal, disk(2))

    if debug:
        save_debug_jpg(
            opened,
            os.path.join(debug_folder, f'{base_name}_step03b_opening.jpg'),
            title='Step 3b: After Binary Opening',
            cmap='gray'
        )

    closed = binary_closing(opened, disk(3))

    if debug:
        save_debug_jpg(
            closed,
            os.path.join(debug_folder, f'{base_name}_step03c_closing.jpg'),
            title='Step 3c: After Binary Closing',
            cmap='gray'
        )

    holes_removed = remove_small_holes(closed, area_threshold=100)

    if debug:
        save_debug_jpg(
            holes_removed,
            os.path.join(debug_folder, f'{base_name}_step03d_holes_removed.jpg'),
            title='Step 3d: Small Holes Removed',
            cmap='gray'
        )

    objects_removed = remove_small_objects(holes_removed, min_size=20)

    if debug:
        save_debug_jpg(
            objects_removed,
            os.path.join(debug_folder, f'{base_name}_step03e_small_objects_removed.jpg'),
            title='Step 3e: Small Objects Removed',
            cmap='gray'
        )

    mask = binary_dilation(objects_removed, disk(2))

    if debug:
        save_debug_jpg(
            mask,
            os.path.join(debug_folder, f'{base_name}_step04_final_morph_mask.jpg'),
            title='Step 4: Final Morphological Mask',
            cmap='gray'
        )

    # 4. Rasterize outfall points and compute distance transform
    outfalls = gpd.read_file(outfall_shapefile).to_crs(crs)

    outfall_r = rasterize(
        [(geom, 1) for geom in outfalls.geometry],
        out_shape=(h, w),
        transform=transform,
        fill=0,
        dtype='uint8'
    )

    if debug:
        save_debug_jpg(
            outfall_r,
            os.path.join(debug_folder, f'{base_name}_step05_outfalls_rasterized.jpg'),
            title='Step 5: Rasterized Outfalls',
            cmap='gray'
        )

    dist_px = distance_transform_edt(outfall_r == 0)

    aoi_geom = wkt.loads(wkt_aoi)
    mid_lat = (aoi_geom.bounds[1] + aoi_geom.bounds[3]) / 2.0

    m_per_lon = 111320 * abs(math.cos(math.radians(mid_lat)))
    pix_deg = abs(transform.a)
    dist_m = dist_px * (pix_deg * m_per_lon)

    if debug:
        save_debug_jpg(
            dist_m,
            os.path.join(debug_folder, f'{base_name}_step06_distance_m.jpg'),
            title='Step 6: Distance from Outfalls, meters',
            cmap='magma'
        )

    # 5. Compute exponential decay probability across image
    prob_full = np.exp(-dist_m / decay_scale)

    if debug:
        save_debug_jpg(
            prob_full,
            os.path.join(debug_folder, f'{base_name}_step07_decay_probability.jpg'),
            title='Step 7: Distance Decay Probability',
            cmap='viridis',
            vmin=0,
            vmax=1
        )

    # 6. Label clusters and assign max probability per cluster
    labels, n_clusters = label(mask)

    if debug:
        save_debug_jpg(
            labels,
            os.path.join(debug_folder, f'{base_name}_step08_labeled_clusters.jpg'),
            title=f'Step 8: Labeled Clusters, n={n_clusters}',
            cmap='tab20'
        )

    prob_map = np.zeros_like(prob_full, dtype=np.float32)
    selected_labels = []

    for c in range(1, n_clusters + 1):
        region = labels == c

        if not np.any(region):
            continue

        max_p = prob_full[region].max()
        prob_map[region] = max_p

        if max_p >= cluster_prob_threshold:
            selected_labels.append(c)

    if debug:
        save_debug_jpg(
            prob_map,
            os.path.join(debug_folder, f'{base_name}_step09_cluster_probability.jpg'),
            title='Step 9: Cluster Max Probability Map',
            cmap='viridis',
            vmin=0,
            vmax=1
        )

    # 7. Create binary mask of selected clusters
    mask_selected = np.isin(labels, selected_labels)

    if debug:
        save_debug_jpg(
            mask_selected,
            os.path.join(debug_folder, f'{base_name}_step10_selected_mask.jpg'),
            title=f'Step 10: Selected Mask, threshold={cluster_prob_threshold}',
            cmap='gray'
        )

    # 8. Write selected-cluster mask to GeoTIFF using source profile
    os.makedirs(os.path.dirname(mask_path), exist_ok=True)

    mask_profile = profile.copy()
    mask_profile.update(driver='GTiff', dtype='uint8', count=1, nodata=0)

    with rasterio.open(mask_path, 'w', **mask_profile) as dst:
        dst.write(mask_selected.astype('uint8'), 1)

    # 9. Write full cluster probability map to GeoTIFF
    prob_profile = profile.copy()
    prob_profile.update(dtype='float32', nodata=0)

    with rasterio.open(prob_path, 'w', **prob_profile) as dst:
        dst.write(prob_map, 1)

    print(f'  Saved mask: {mask_path}')
    print(f'  Saved prob map: {prob_path}')

    if debug:
        print(f'  Saved debug JPGs to: {debug_folder}')


def batch_process(folder, output_root, wkt_aoi, threshold_value,
                  outfall_shapefile, decay_scale, cluster_prob_threshold,
                  debug=False):

    os.makedirs(output_root, exist_ok=True)

    debug_folder = os.path.join(output_root, 'debug_jpgs')

    for root, _, files in os.walk(folder):
        for file in files:
            if file.startswith('.') or not file.endswith('_cumulative.tif'):
                continue

            inp = os.path.join(root, file)
            base = os.path.splitext(file)[0]

            mask_out = os.path.join(output_root, f'{base}_mask.tif')
            prob_out = os.path.join(output_root, f'{base}_prob.tif')

            process_image(
                inp,
                mask_out,
                prob_out,
                wkt_aoi,
                threshold_value,
                outfall_shapefile,
                decay_scale,
                cluster_prob_threshold,
                debug=debug,
                debug_folder=debug_folder
            )


if __name__ == '__main__':
    batch_process(
        input_folder,
        output_folder,
        wkt_aoi,
        threshold_value,
        outfall_shapefile,
        decay_scale,
        cluster_prob_threshold,
        debug=DEBUG
    )

    print('Batch complete.')