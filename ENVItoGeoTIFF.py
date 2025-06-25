import os
import argparse
import numpy as np
from spectral import open_image
from osgeo import gdal, osr

REFERENCE_PATH = "/Users/ereilly/Documents/code/0619_lpass_iseg.tif"

def convert_envi_to_geotiff(envi_hdr_path, output_tif_path):
    img = open_image(envi_hdr_path)
    data = img.read_band(0)

    driver = gdal.GetDriverByName('GTiff')
    height, width = data.shape
    out_ds = driver.Create(output_tif_path, width, height, 1, gdal.GDT_Int16)
    out_ds.GetRasterBand(1).WriteArray(data)

    # Apply geotransform and projection from fixed reference
    if os.path.exists(REFERENCE_PATH):
        ref_ds = gdal.Open(REFERENCE_PATH)
        if ref_ds:
            out_ds.SetGeoTransform(ref_ds.GetGeoTransform())
            out_ds.SetProjection(ref_ds.GetProjection())
            print(f"↪ Applied geotransform and projection from reference: {REFERENCE_PATH}")
        else:
            print(f"⚠️ Failed to open reference file. Forcing WGS84 projection.")
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(4326)
            out_ds.SetProjection(srs.ExportToWkt())
    else:
        print(f"⚠️ Reference file not found: {REFERENCE_PATH}")
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        out_ds.SetProjection(srs.ExportToWkt())

    out_ds.FlushCache()
    out_ds = None
    print(f"✓ Saved GeoTIFF: {output_tif_path}")

def batch_convert(base_dir):
    for root, dirs, files in os.walk(base_dir):
        for fname in files:
            if fname.endswith(".dat.hdr"):
                hdr_path = os.path.join(root, fname)
                dat_path = hdr_path.replace(".hdr", "")
                tif_out = dat_path.replace(".dat", ".tif")

                try:
                    convert_envi_to_geotiff(hdr_path, tif_out)
                except Exception as e:
                    print(f"⚠️ Failed to convert {hdr_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch convert ENVI .dat/.hdr files to GeoTIFF with WGS84 projection")
    parser.add_argument("folder", help="Top-level folder containing subfolders with .dat/.hdr outputs")
    args = parser.parse_args()

    batch_convert(args.folder)
