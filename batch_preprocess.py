from esa_snappy import ProductIO, GPF, HashMap, jpy, ProductUtils
import os
import math
import rasterio
from rasterio import features
import geopandas as gpd
import numpy as np
from scipy.ndimage import distance_transform_edt

# --- Configuration ---
input_folder       = '/Volumes/External/TJ_SAR/01_data/02_2025_2020'
output_folder      = '/Volumes/External/TJ_SAR/02_preprocessed/02_20202025'
shapefile_path     = '/Volumes/External/TJ_SAR/01_data/shapefiles/SanDiegoBay.shp'
wkt = "POLYGON ((-117.210388 32.379961, -117.059326 32.379961, -117.059326 32.640531, -117.210388 32.640531, -117.210388 32.379961)))"
distance_threshold = 5000  # meters from any shoreline

# --- SNAP Workflow ---
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
    HashMapType   = jpy.get_type('java.util.HashMap')
    BandDescriptor= jpy.get_type('org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor')
    band_array    = jpy.array('org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor', 1)
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
    sources = HashMapType(); sources.put('master', product); sources.put('slave', incang)
    return GPF.createProduct('BandMerge', merge_params, sources)

def ellipsoid_correction(product, proj='WGS84(DD)'):
    params = HashMap()
    params.put('sourceBands', ",".join(product.getBandNames()))
    params.put('imgResamplingMethod', 'BILINEAR_INTERPOLATION')
    params.put('mapProjection', proj)
    return GPF.createProduct('Ellipsoid-Correction-GG', params, product)

def apply_land_sea_mask(product):
    HashMapType = jpy.get_type('java.util.HashMap')
    JInteger    = jpy.get_type('java.lang.Integer')
    params = HashMapType()
    params.put('landMask', True)
    params.put('useSRTM', True)
    params.put('shorelineExtension', JInteger(4))
    return GPF.createProduct('Land-Sea-Mask', params, product)

def subset_to_aoi(product, wkt_string):
    params = HashMap()
    params.put('geoRegion', wkt_string)
    params.put('copyMetadata', True)
    return GPF.createProduct('Subset', params, product)

def reorder_bands_explicitly(product, order):
    PD   = jpy.get_type('org.esa.snap.core.datamodel.ProductData')
    BT   = jpy.get_type('org.esa.snap.core.datamodel.Band')
    PDAT = jpy.get_type('org.esa.snap.core.datamodel.Product')
    PU   = jpy.get_type('org.esa.snap.core.util.ProductUtils')
    w, h = product.getSceneRasterWidth(), product.getSceneRasterHeight()
    new  = PDAT('Ordered','GeoTIFF', w, h)
    PU.copyGeoCoding(product, new)
    PU.copyMetadata(product, new)
    PU.copyTiePointGrids(product, new)
    for name in order:
        src_band = product.getBand(name)
        if src_band is None:
            raise ValueError(f"Band {name} missing")
        tgt = BT(name, src_band.getDataType(), w, h)
        new.addBand(tgt)
        pix = src_band.readPixels(0, 0, w, h, jpy.array('float', w*h))
        ras = PD.createInstance(PD.TYPE_FLOAT32, w*h)
        for i, val in enumerate(pix):
            ras.setElemFloatAt(i, val)
        tgt.setRasterData(ras)
    return new

# --- Combined Mask Function with correct meter conversion ---
def mask_with_shapefile_and_5km(raster_path, shapefile_path,
                                 distance_threshold, output_path):
    # Bay mask for excluding inside bay
    bay_gdf = gpd.read_file(shapefile_path).to_crs(epsg=4326)

    with rasterio.open(raster_path) as src:
        meta   = src.meta.copy()
        data   = src.read().astype('float32')
        nodata = src.nodata if src.nodata is not None else np.nan

        # Convert zeros to nodata
        data[data == 0] = nodata

        # Compute water mask (non-nan pixels)
        water_mask = ~np.isnan(data[0])

        # Distance in pixels to nearest land (nan)
        dist_pix = distance_transform_edt(water_mask)

        # Approximate pixel size in meters by converting degrees to meters at mid-latitude
        bounds = src.bounds  # left, bottom, right, top in lon/lat
        mid_lat = (bounds.top + bounds.bottom) / 2.0
        # meters per degree longitude at mid-latitude
        m_per_deg = 111320 * abs(math.cos(math.radians(mid_lat)))
        pix_deg  = abs(src.transform.a)
        pix_m    = pix_deg * m_per_deg

        dist_m = dist_pix * pix_m

        # Debug prints
        print(f"[DEBUG] Pixel size: {pix_deg:.6f}° ≈ {pix_m:.2f} m")
        print(f"[DEBUG] Distances (m) > min {np.nanmin(dist_m):.2f}, max {np.nanmax(dist_m):.2f}")

        # Mask inside bay polygon → nodata
        mask_bay = features.geometry_mask(
            [geom for geom in bay_gdf.geometry],
            out_shape=(src.height, src.width),
            transform=src.transform,
            invert=True
        )
        data[:, mask_bay] = nodata

        # Mask out beyond threshold → nodata
        far_mask = dist_m > distance_threshold
        data[:, far_mask] = nodata

        meta.update(dtype=rasterio.float32, nodata=nodata)

    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(data)

    print(f"Saved masked & distance-clipped raster to: {output_path}")

# --- Scene Processing ---
def write_product(prod, path, fmt='GeoTIFF'):
    ProductIO.writeProduct(prod, path, fmt)

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
    p7 = reorder_bands_explicitly(p6, ['Sigma0_VV', 'incang'])

    tmp = '/tmp/temp_sar.tif'
    write_product(p7, tmp)
    mask_with_shapefile_and_5km(tmp, shapefile_path, distance_threshold, outp)

    if os.path.exists(tmp): os.remove(tmp)

# --- Main ---
if __name__ == '__main__':
    initialize_snap()
    os.makedirs(output_folder, exist_ok=True)
    zip_files = [f for f in os.listdir(input_folder)
                if f.endswith('.zip') and not f.startswith('._')]
    print(f"Found {len(zip_files)} zip files")
    for fname in zip_files:
        inp = os.path.join(input_folder, fname)
        out = os.path.join(
            output_folder,
            os.path.splitext(fname)[0] + '_pre.tif'
        )
        try:
            process_scene(inp, out)
        except Exception as e:
            print(f"Error processing {fname}: {e}")
    print("All done!")


print("◝(ᵔᗜᵔ)◜ done!! yayy!!")