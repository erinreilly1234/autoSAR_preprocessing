import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
import rasterio.features
import geopandas as gpd
import math
from shapely.geometry import shape
from datetime import datetime
from pytz import timezone, utc
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, Normalize
import matplotlib.patches as patches

# --- Parameters ---
original_root = "/Volumes/External/TJ_SAR/02_preprocessed/02_20202025"
mask_root = "/Volumes/External/TJ_SAR/04_iseg"
output_plot_dir = "/Volumes/External/TJ_SAR/05_viz"
output_shapefile_dir = "/Volumes/External/TJ_SAR/06_shapefiles"  # new shapefile output
met_csv_path = "/Volumes/External/TJ_estuary/analysis/TJRTLMET.csv"
threshold = 0  # only keep mask values > threshold

# timezones
pacific = timezone("US/Pacific")

# --- Load meteorological data (in Pacific Time) ---
met_df = pd.read_csv(met_csv_path)
met_df.columns = met_df.columns.str.strip()
met_df["DateTimeStamp"] = pd.to_datetime(
    met_df["DateTimeStamp"], format="%m/%d/%y %H:%M", errors="coerce"
)
met_df["DateTimeStamp"] = met_df["DateTimeStamp"].dt.tz_localize(
    pacific,
    ambiguous="NaT",
    nonexistent="NaT"
)
met_df = met_df.dropna(subset=["DateTimeStamp"]).sort_values("DateTimeStamp")

# --- Utility functions ---

def extract_datetime_from_filename(filename):
    """Extract UTC datetime from Sentinel-1 filename."""
    parts = filename.split("_")
    for part in parts:
        if part.startswith("20") and "T" in part:
            try:
                return utc.localize(datetime.strptime(part, "%Y%m%dT%H%M%S"))
            except ValueError:
                continue
    return None


def get_nearest_met_data(local_time):
    """Find the closest met record to the given local time."""
    diffs = (met_df["DateTimeStamp"] - local_time).abs()
    idx = diffs.idxmin()
    row = met_df.loc[idx]
    return row["WSpd"], row["Wdir"], row["DateTimeStamp"]


def match_original_path(mask_filename):
    return os.path.join(
        original_root,
        mask_filename.replace("_JPL0.4_VVDR_cumulative_mask.tif", ".tif")
    )

# --- Main processing loop ---
# make sure output dirs exist
os.makedirs(output_plot_dir, exist_ok=True)
os.makedirs(output_shapefile_dir, exist_ok=True)

for root, _, files in os.walk(mask_root):
    for file in files:
        if file.startswith('.') or not file.endswith("_JPL0.4_VVDR_cumulative_mask.tif"):
            continue

        mask_path = os.path.join(root, file)
        original_path = match_original_path(file)

        if not os.path.exists(original_path):
            print(f"[!] Original image not found for: {file}")
            continue

        # --- Read mask and prepare shapefile export ---
        with rasterio.open(mask_path) as src_mask:
            mask_arr = src_mask.read(1)
            transform = src_mask.transform
            crs = src_mask.crs
            nodata = src_mask.nodata

        # Build boolean mask: pixels > threshold and not nodata
        valid_mask = mask_arr > threshold
        if nodata is not None:
            valid_mask &= (mask_arr != nodata)

        # Vectorize mask to shapes
        shapes = []
        for geom, val in rasterio.features.shapes(
            mask_arr.astype('int16'),
            mask=valid_mask,
            transform=transform
        ):
            shapes.append({'geometry': shape(geom), 'value': int(val)})

        # Write GeoDataFrame if any shapes found
        if shapes:
            gdf = gpd.GeoDataFrame(shapes, crs=crs)
            # optional: dissolve into one multipart if only total area needed
            # gdf = gdf.dissolve(by='value')
            shp_name = file.replace(
                "_JPL0.4_VVDR_cumulative_mask.tif", "_mask.shp"
            )
            out_shp = os.path.join(output_shapefile_dir, shp_name)
            gdf.to_file(out_shp, driver='ESRI Shapefile')
            print(f"  → wrote shapefile: {out_shp}")
        else:
            print(f"  → no valid mask pixels for: {file}, skipping shapefile.")

        # --- Read original for plotting and get meteorology ---
        image_time_utc = extract_datetime_from_filename(file)
        image_time_local = image_time_utc.astimezone(pacific)
        wspd, wdir, met_time = get_nearest_met_data(image_time_local)

        with rasterio.open(original_path) as src_orig:
            original = src_orig.read(1).astype(float)
            bounds = src_orig.bounds
            extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

        # Stretch original reflectance
        vmin, vmax = np.nanpercentile(original, (2, 95))
        clipped = np.clip(original, vmin, vmax)
        normed = (clipped - vmin) / (vmax - vmin)

        # Build RGBA overlay
        overlay = np.zeros((mask_arr.shape[0], mask_arr.shape[1], 4))
        overlay[valid_mask] = [1.0, 1.0, 0.8, 1.0]

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(9, 12), constrained_layout=True)
        axes[0].imshow(normed, cmap='gray', extent=extent)
        axes[0].set_title('Original (2–98% Stretch)', fontsize=12)
        axes[0].axis('off')

        axes[1].imshow(normed, cmap='gray', extent=extent)
        axes[1].imshow(overlay, extent=extent)
        axes[1].set_title('Original with Plume Mask', fontsize=12)
        axes[1].axis('off')

        # Annotation and wind vector
        wind_text = (
            f"Wind: {wspd:.1f} m/s @ {wdir:.0f}°\n"
            f"Local Time: {image_time_local.strftime('%Y-%m-%d %H:%M')}"
        )
        fig.text(0.5, 0.05, wind_text, ha='center', fontsize=13, fontweight='bold')
        wind_angle_deg = (270 - wdir) % 360
        angle_rad = math.radians(wind_angle_deg)
        height = bounds.top - bounds.bottom
        base_frac = 0.07
        scale = np.clip(wspd / 10, 0.5, 1.5)
        arrow_length = height * base_frac * scale
        dx = arrow_length * math.cos(angle_rad)
        dy = arrow_length * math.sin(angle_rad)
        x0 = bounds.right - 0.1 * (bounds.right - bounds.left)
        y0 = bounds.bottom + 0.05 * height
        axes[1].add_patch(
            patches.FancyArrow(
                x0, y0, dx, dy,
                width=arrow_length * 0.05,
                head_width=arrow_length * 0.15,
                head_length=arrow_length * 0.15,
                length_includes_head=True,
                transform=axes[1].transData,
                color='lime',
                alpha=0.9
            )
        )

        # Save figure
        flat_name = file.replace(
            "_JPL0.4_VVDR_cumulative_mask.tif", "_overlay.png"
        )
        out_png = os.path.join(output_plot_dir, flat_name)
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        plt.savefig(out_png, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  → saved plot: {out_png}")

print("Done!")
