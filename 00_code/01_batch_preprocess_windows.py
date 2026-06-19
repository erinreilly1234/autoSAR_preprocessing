"""
Windows-safe Sentinel-1 batch processor for SNAP (snappy)
"""

from esa_snappy import jpy, ProductIO, GPF, HashMap, ProductUtils
import os
import sys
import subprocess
import math
import rasterio
from rasterio import features
import geopandas as gpd
import numpy as np
from scipy.ndimage import distance_transform_edt
import zipfile


# ------------------------------------------------------------------------------
# WINDOWS-SAFE PATHS
# ------------------------------------------------------------------------------
input_folder      = r"E:/TJ/010_rawdata/test/test_data"
output_folder     = r"E:/TJ/020_preprocessing/sentinel1"
shapefile_path    = r"E:/TJ_SAR/01_data/shapefiles/SanDiegoBay.shp"

# Force SNAP & Python temporary directory on Windows
WINDOWS_TMP = r"C:/temp_sar"
os.makedirs(WINDOWS_TMP, exist_ok=True)

os.environ["TMP"] = WINDOWS_TMP
os.environ["TEMP"] = WINDOWS_TMP
os.environ["TMPDIR"] = WINDOWS_TMP


# ------------------------------------------------------------------------------
# OTHER CONFIG
# ------------------------------------------------------------------------------
wkt = ("POLYGON ((-117.457581 32.268555, -117.007141 32.268555, "
       "-117.007141 32.724909, -117.457581 32.724909, "
       "-117.457581 32.268555)))")

distance_threshold = 4000
apply_distance_mask = False
apply_shapefile_mask = False


# ------------------------------------------------------------------------------
# SNAP JVM SETTINGS
# ------------------------------------------------------------------------------
try:
    jpy.create_jvm([
        "-Xms1g",
        "-Xmx12g",
        "-Djava.awt.headless=true",
    ])
except Exception:
    pass


# ------------------------------------------------------------------------------
# SNAP PROCESSING FUNCTIONS
# ------------------------------------------------------------------------------

def initialize_snap():
    GPF.getDefaultInstance().getOperatorSpiRegistry().loadOperatorSpis()

def load_product(path):
    return ProductIO.readProduct(path)

def calibrate_product(product):
    params = HashMap()
    params.put('outputSigmaBand', True)
    params.put('selectedPolarisations', 'VV')
    params.put('sourceBands', 'Intensity_VV')
    params.put('auxFile', 'Latest Auxiliary File')
    params.put('outputImageInComplex', False)
    return GPF.createProduct('Calibration', params, product)

def apply_multilook(product, rg_looks=2, az_looks=2):
    JInteger = jpy.get_type('java.lang.Integer')
    params = HashMap()
    params.put('nRgLooks', JInteger(rg_looks))
    params.put('nAzLooks', JInteger(az_looks))
    params.put('outputIntensity', True)
    return GPF.createProduct('Multilook', params, product)

def add_incang_band(product):
    HashMapType    = jpy.get_type('java.util.HashMap')
    BandDescriptor = jpy.get_type(
        'org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor'
    )
    band_array = jpy.array(
        'org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor', 1
    )
    desc = BandDescriptor()
    desc.name       = 'incang'
    desc.type       = 'float32'
    desc.expression = 'incident_angle'
    band_array[0]   = desc
    params = HashMapType()
    params.put('targetBands', band_array)
    params.put('retainExistingBands', False)
    incang = GPF.createProduct('BandMaths', params, product)

    merge_params = HashMapType()
    merge_params.put('sourceProductNames', 'master,slave')
    merge_params.put('resamplingMethod', 'NEAREST_NEIGHBOUR')
    merge_params.put('geodeticTiePoints', True)
    sources = HashMapType()
    sources.put('master', product)
    sources.put('slave', incang)
    return GPF.createProduct('BandMerge', merge_params, sources)

def ellipsoid_correction(product, proj='WGS84(DD)'):
    params = HashMap()
    params.put('sourceBands', ",".join(product.getBandNames()))
    params.put('imgResamplingMethod', 'BILINEAR_INTERPOLATION')
    params.put('mapProjection', proj)
    return GPF.createProduct('Ellipsoid-Correction-GG', params, product)

def apply_land_sea_mask(product):
    params = HashMap()
    params.put('landMask', True)
    params.put('useSRTM', True)
    params.put('shorelineExtension', jpy.get_type('java.lang.Integer')(4))
    return GPF.createProduct('Land-Sea-Mask', params, product)

def subset_to_aoi(product, wkt_string):
    params = HashMap()
    params.put('geoRegion', wkt_string)
    params.put('copyMetadata', True)
    return GPF.createProduct('Subset', params, product)


# ------------------------------------------------------------------------------
# WINDOWS-SAFE TEMP FILE WRITING
# ------------------------------------------------------------------------------
def write_product_windows(prod):
    out = os.path.join(WINDOWS_TMP, "temp_sar.tif")
    ProductIO.writeProduct(prod, out, 'GeoTIFF')
    return out


# ------------------------------------------------------------------------------
# MASKING (unchanged)
# ------------------------------------------------------------------------------
def mask_with_shapefile_and_5km(raster_path, shapefile_path, distance_threshold, output_path):

    global apply_distance_mask, apply_shapefile_mask

    if apply_shapefile_mask:
        bay_gdf = gpd.read_file(shapefile_path).to_crs(epsg=4326)

    with rasterio.open(raster_path) as src:
        meta   = src.meta.copy()
        data   = src.read().astype('float32')
        nodata = src.nodata if src.nodata is not None else np.nan
        data[data == 0] = nodata

        water_mask = ~np.isnan(data[0])
        dist_pix = distance_transform_edt(water_mask)

        bounds = src.bounds
        mid_lat = (bounds.top + bounds.bottom) / 2.0
        m_per_deg = 111320 * abs(math.cos(math.radians(mid_lat)))
        pix_deg = abs(src.transform.a)
        pix_m = pix_deg * m_per_deg
        dist_m = dist_pix * pix_m

        if apply_shapefile_mask:
            mask_bay = features.geometry_mask(
                bay_gdf.geometry,
                out_shape=(src.height, src.width),
                transform=src.transform,
                invert=True
            )
            data[:, mask_bay] = nodata

        if apply_distance_mask:
            far_mask = dist_m > distance_threshold
            data[:, far_mask] = nodata

        meta.update(dtype=rasterio.float32, nodata=nodata)

    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(data)

    print(f"[INFO] Saved {output_path}")


# ------------------------------------------------------------------------------
# MAIN SCENE PIPELINE
# ------------------------------------------------------------------------------
def process_scene(inp, outp):

    basename = os.path.splitext(os.path.basename(inp))[0]
    print(f"Processing: {basename}")

    p0 = load_product(inp)
    p1 = calibrate_product(p0)
    ProductUtils.copyTiePointGrids(p0, p1)
    p2 = apply_multilook(p1)
    p3 = add_incang_band(p2)
    p4 = ellipsoid_correction(p3)
    p5 = apply_land_sea_mask(p4)
    p6 = subset_to_aoi(p5, wkt)

    tmp = write_product_windows(p6)
    mask_with_shapefile_and_5km(tmp, shapefile_path, distance_threshold, outp)


# ------------------------------------------------------------------------------
# MASTER ENTRYPOINT
# ------------------------------------------------------------------------------
if __name__ == "__main__":

    args = sys.argv[1:]

    # -------------------------
    # BATCH MODE
    # -------------------------
    if len(args) == 0:

        zip_files = []
        for f in os.listdir(input_folder):
            f = f.lstrip("/\\")       # Windows-safe
            if f.endswith(".zip"):
                zip_files.append(os.path.join(input_folder, f))

        print(f"Found {len(zip_files)} zip files")

        for z in zip_files:
            if not zipfile.is_zipfile(z):
                print(f"Skipping invalid zip: {z}")
                continue
            print(f"Spawning: {z}")
            subprocess.call([sys.executable, __file__, z])

        print("Batch complete")
        sys.exit(0)

    # -------------------------
    # SINGLE MODE
    # -------------------------
    elif len(args) == 1:
        inp = args[0]
        outp = os.path.join(output_folder,
                            os.path.splitext(os.path.basename(inp))[0] + "_pre.tif")

        os.makedirs(output_folder, exist_ok=True)

        initialize_snap()
        process_scene(inp, outp)
        sys.exit(0)

    else:
        print("Usage: python script.py [optional_single_zip_path]")
        sys.exit(1)
