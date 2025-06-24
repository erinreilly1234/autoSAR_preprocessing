from esa_snappy import ProductIO, GPF, HashMap, jpy, ProductUtils
import os
import rasterio
from rasterio import features  # <-- this is the missing import
import geopandas as gpd
import numpy as np

# --- SNAP Workflow Functions ---
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
    tie_grids = [product.getTiePointGridAt(i).getName() for i in range(product.getNumTiePointGrids())]
    if 'incident_angle' not in tie_grids:
        raise RuntimeError("'incident_angle' tie-point grid not found.")

    HashMap = jpy.get_type('java.util.HashMap')
    BandDescriptor = jpy.get_type('org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor')
    band_array = jpy.array('org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor', 1)

    descriptor = BandDescriptor()
    descriptor.name = 'incang'
    descriptor.type = 'float32'
    descriptor.expression = 'incident_angle'
    band_array[0] = descriptor

    params = HashMap()
    params.put('targetBands', band_array)
    params.put('retainExistingBands', False)

    incang_product = GPF.createProduct('BandMaths', params, product)

    merge_params = HashMap()
    merge_params.put('sourceProductNames', 'master,slave')
    merge_params.put('resamplingMethod', 'NEAREST_NEIGHBOUR')
    merge_params.put('geodeticTiePoints', True)

    sources = HashMap()
    sources.put('master', product)
    sources.put('slave', incang_product)

    return GPF.createProduct('BandMerge', merge_params, sources)

def ellipsoid_correction(product, proj='WGS84(DD)'):
    band_names = list(product.getBandNames())
    params = HashMap()
    params.put('sourceBands', ",".join(band_names))
    params.put('imgResamplingMethod', 'BILINEAR_INTERPOLATION')
    params.put('mapProjection', proj)
    return GPF.createProduct('Ellipsoid-Correction-GG', params, product)

def apply_land_sea_mask(product):
    HashMap = jpy.get_type('java.util.HashMap')
    JInteger = jpy.get_type('java.lang.Integer')
    params = HashMap()
    params.put('landMask', True)
    params.put('useSRTM', True)
    params.put('shorelineExtension', JInteger(2))
    return GPF.createProduct('Land-Sea-Mask', params, product)

def subset_to_aoi(product, wkt_string):
    HashMap = jpy.get_type('java.util.HashMap')
    params = HashMap()
    params.put('geoRegion', wkt_string)
    params.put('copyMetadata', True)
    return GPF.createProduct('Subset', params, product)

def reorder_bands_explicitly(product, output_band_order):
    ProductData = jpy.get_type('org.esa.snap.core.datamodel.ProductData')
    BandType = jpy.get_type('org.esa.snap.core.datamodel.Band')
    Product = jpy.get_type('org.esa.snap.core.datamodel.Product')
    ProductUtils = jpy.get_type('org.esa.snap.core.util.ProductUtils')

    width = product.getSceneRasterWidth()
    height = product.getSceneRasterHeight()

    new_product = Product("OrderedProduct", "GeoTIFF", width, height)

    ProductUtils.copyGeoCoding(product, new_product)
    ProductUtils.copyMetadata(product, new_product)
    ProductUtils.copyTiePointGrids(product, new_product)

    for band_name in output_band_order:
        source_band = product.getBand(band_name)
        if source_band is None:
            raise ValueError(f"Band '{band_name}' not found in product.")
        band_type = source_band.getDataType()
        new_band = BandType(band_name, band_type, width, height)
        new_product.addBand(new_band)

        pixel_data = source_band.readPixels(0, 0, width, height, jpy.array('float', width * height))
        raster = ProductData.createInstance(ProductData.TYPE_FLOAT32, width * height)
        for i in range(len(pixel_data)):
            raster.setElemFloatAt(i, pixel_data[i])
        new_band.setRasterData(raster)

    return new_product

def mask_raster_with_shapefile(raster_path, shapefile_path, output_path):
    """
    Masks all pixels inside the shapefile geometry in the raster and sets them to nodata.
    """
    gdf = gpd.read_file(shapefile_path)
    gdf = gdf.to_crs(epsg=4326)  # Adjust CRS if needed

    with rasterio.open(raster_path) as src:
        out_meta = src.meta.copy()
        out_image = src.read()

        mask = features.geometry_mask(
            [geom for geom in gdf.geometry],
            out_shape=(src.height, src.width),
            transform=src.transform,
            invert=True
        )

        for i in range(out_image.shape[0]):
            out_image[i, mask] = src.nodata if src.nodata is not None else 0

    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(out_image)

    print(f"Masked raster saved to: {output_path}")

def write_product(product, output_path, format='GeoTIFF'):
    ProductIO.writeProduct(product, output_path, format)

def process_scene(input_path, output_path):
    print(f"Processing: {input_path}")
    product = load_product(input_path)
    calibrated = calibrate_product(product)
    ProductUtils.copyTiePointGrids(product, calibrated)
    multilooked = apply_multilook(calibrated)
    multilooked = add_incang_band(multilooked)
    geocoded = ellipsoid_correction(multilooked)
    masked = apply_land_sea_mask(geocoded)

    wkt_aoi = "POLYGON ((-117.171936 32.379961, -117.055206 32.379961, -117.055206 32.638218, -117.171936 32.638218, -117.171936 32.379961)))"
    subset = subset_to_aoi(masked, wkt_aoi)

    ordered = reorder_bands_explicitly(subset, ['Sigma0_VV', 'incang'])
    # Write to a temporary path for masking
    temp_output_path = "/tmp/temp_output.tif"  # or use tempfile.NamedTemporaryFile()

    print("Writing temporary GeoTIFF for masking...")
    write_product(ordered, temp_output_path)

    masked_output_path = f"/Users/ereilly/Documents/code/autoSAR_preprocessing/test/output/{basename}_pre.tif"
    print("Masking with shapefile...")
    mask_raster_with_shapefile(temp_output_path, shapefile_path, masked_output_path)

    # Optionally delete the temp file if desired
    if os.path.exists(temp_output_path):
        os.remove(temp_output_path)
    print("Process complete.")

# --- Batch Loop ---
input_folder = '/Volumes/External/TJ_estuary/01_data/sentinel_1/01_JunethroughDec'
output_folder = '/Users/ereilly/Documents/code/autoSAR_preprocessing/test/output'
shapefile_path = '/Volumes/External/TJ_estuary/visualization/SARplume/_data/SanDiegoBay.shp'

if __name__ == '__main__':
    initialize_snap()
    zip_files = [f for f in os.listdir(input_folder) if f.endswith('.zip') and not f.startswith('._')]
    print(f"Found {len(zip_files)} zip files to process.")

    for fname in zip_files:
        input_path = os.path.join(input_folder, fname)
        basename = os.path.splitext(fname)[0]
        output_path = os.path.join(output_folder, f"{basename}_pre.tif")

        try:
            process_scene(input_path, output_path)
        except Exception as e:
            print(f"Error processing {fname}: {e}")