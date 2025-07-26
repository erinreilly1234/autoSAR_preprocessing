import os
import math
import numpy as np
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
input_folder = '/Volumes/External/TJ_SAR/03_oilClassification/ContrastRatio/TJ'
output_folder = '/Volumes/External/TJ_SAR/04_iseg/TJ'
wkt_aoi = (
    'POLYGON ((-117.171936 32.379961, -117.055206 32.379961,'
    ' -117.055206 32.638218, -117.171936 32.638218, -117.171936 32.379961))'
)
threshold_value = 0.11       # contrast ratio threshold for initial oil detection
outfall_shapefile = '/Volumes/External/TJ_SAR/01_data/shapefiles/Outflow.shp'
decay_scale = 4000.0         # distance decay scale (meters)
cluster_prob_threshold = 0.72  # minimum probability threshold for cluster acceptance


def process_image(image_path, mask_path, prob_path,
                  wkt_aoi, threshold_value,
                  outfall_shapefile, decay_scale, cluster_prob_threshold):
    # 1. Open the source GeoTIFF to get data, metadata, and geotransform
    with rasterio.open(image_path) as src:
        img = src.read(1).astype(np.float32)
        profile = src.profile
        transform = src.transform
        crs = src.crs
        h, w = src.height, src.width

    # 2. Low-pass filter with valid-data weighting
    valid = (~np.isnan(img)).astype(np.float32)
    filled = np.nan_to_num(img, nan=0.0)
    num = uniform_filter(filled, size=3, mode='nearest')
    den = uniform_filter(valid, size=3, mode='nearest')
    smooth = np.full_like(img, np.nan, dtype=np.float32)
    mask_den = den > 0
    smooth[mask_den] = num[mask_den] / den[mask_den]

    # 3. Threshold & morphological cleanup
    signal = (smooth > threshold_value) & ~np.isnan(smooth)
    clean  = binary_opening(signal, disk(2))
    clean  = binary_closing(clean, disk(3))
    clean  = remove_small_holes(clean, area_threshold=300)
    clean  = remove_small_objects(clean, min_size=100)
    mask   = binary_dilation(clean, disk(2))

    # 4. Rasterize outfall points and compute distance transform
    outfalls = gpd.read_file(outfall_shapefile).to_crs(crs)
    outfall_r = rasterize(
        [(geom, 1) for geom in outfalls.geometry],
        out_shape=(h, w), transform=transform,
        fill=0, dtype='uint8'
    )
    dist_px = distance_transform_edt(outfall_r == 0)
    mid_lat = (wkt.loads(wkt_aoi).bounds[1] + wkt.loads(wkt_aoi).bounds[3]) / 2.0
    m_per_lon = 111320 * abs(math.cos(math.radians(mid_lat)))
    pix_deg   = abs(transform.a)
    dist_m    = dist_px * (pix_deg * m_per_lon)

    # 5. Compute exponential decay probability across image
    prob_full = np.exp(-dist_m / decay_scale)

    # 6. Label clusters and assign max probability per cluster
    labels, n_clusters = label(mask)
    prob_map = np.zeros_like(prob_full, dtype=np.float32)
    selected_labels = []
    for c in range(1, n_clusters + 1):
        region = (labels == c)
        if not np.any(region):
            continue
        max_p = prob_full[region].max()  # highest decay probability in cluster
        # assign max probability for all clusters
        prob_map[region] = max_p
        # threshold only for selection mask
        if max_p >= cluster_prob_threshold:
            selected_labels.append(c)

    # 7. Create binary mask of selected clusters
    mask_selected = np.isin(labels, selected_labels)

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


def batch_process(folder, output_root, wkt_aoi, threshold_value,
                  outfall_shapefile, decay_scale, cluster_prob_threshold):
    os.makedirs(output_root, exist_ok=True)
    for root, _, files in os.walk(folder):
        for file in files:
            if file.startswith('.') or not file.endswith('_cumulative.tif'):
                continue
            inp = os.path.join(root, file)
            base = os.path.splitext(file)[0]
            mask_out = os.path.join(output_root, f'{base}_mask.tif')
            prob_out = os.path.join(output_root, f'{base}_prob.tif')
            process_image(
                inp, mask_out, prob_out,
                wkt_aoi, threshold_value,
                outfall_shapefile, decay_scale,
                cluster_prob_threshold
            )

if __name__ == '__main__':
    batch_process(
        input_folder, output_folder, wkt_aoi,
        threshold_value, outfall_shapefile,
        decay_scale, cluster_prob_threshold
    )

print('Batch complete.')


