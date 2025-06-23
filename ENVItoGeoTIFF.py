import os
import argparse
from spectral import open_image
from osgeo import gdal
import numpy as np

def convert_envi_to_geotiff(envi_hdr_path, output_tif_path, reference_path=None):
    img = open_image(envi_hdr_path)
    data = img.read_band(0)

    driver = gdal.GetDriverByName('GTiff')
    height, width = data.shape
    out_ds = driver.Create(output_tif_path, width, height, 1, gdal.GDT_Int16)
    out_ds.GetRasterBand(1).WriteArray(data)

    if reference_path and os.path.exists(reference_path):
        ref_ds = gdal.Open(reference_path)
        if ref_ds:
            out_ds.SetGeoTransform(ref_ds.GetGeoTransform())
            out_ds.SetProjection(ref_ds.GetProjection())

    out_ds.FlushCache()
    out_ds = None
    print(f"✓ Saved GeoTIFF: {output_tif_path}")

def batch_convert(base_dir, use_reference=False):
    for root, dirs, files in os.walk(base_dir):
        for fname in files:
            if fname.endswith(".dat.hdr"):
                hdr_path = os.path.join(root, fname)
                dat_path = hdr_path.replace(".hdr", "")
                tif_out = dat_path.replace(".dat", ".tif")

                reference = None
                if use_reference:
                    # Try to find the original contrast ratio image in the same dir
                    for candidate in os.listdir(root):
                        if "contrast_ratio" in candidate and candidate.endswith(".tif"):
                            reference = os.path.join(root, candidate)
                            break

                try:
                    convert_envi_to_geotiff(hdr_path, tif_out, reference)
                except Exception as e:
                    print(f"⚠️ Failed to convert {hdr_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch convert ENVI .dat/.hdr files to GeoTIFF")
    parser.add_argument("folder", help="Top-level folder containing subfolders with .dat/.hdr outputs")
    parser.add_argument("--use-reference", action="store_true",
                        help="Try to copy geotransform/projection from contrast_ratio .tif files in the same folder")
    args = parser.parse_args()

    batch_convert(args.folder, use_reference=args.use_reference)
