from esa_snappy import ProductIO, GPF, HashMap, jpy, ProductUtils
import os

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
    print("Adding 'incang' band from 'incident_angle'...")

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

    merged = GPF.createProduct('BandMerge', merge_params, sources)
    print("'incang' band successfully merged.")
    return merged

def ellipsoid_correction(product, proj='WGS84(DD)'):
    print("Applying ellipsoid-based geocoding...")

    band_names = list(product.getBandNames())
    print("Bands to geocode:", band_names)

    params = HashMap()
    params.put('sourceBands', ",".join(band_names))
    params.put('imgResamplingMethod', 'BILINEAR_INTERPOLATION')
    params.put('mapProjection', proj)

    return GPF.createProduct('Ellipsoid-Correction-GG', params, product)

def apply_land_sea_mask(product):
    print("Applying Land-Sea Mask...")

    HashMap = jpy.get_type('java.util.HashMap')
    JInteger = jpy.get_type('java.lang.Integer')

    params = HashMap()
    params.put('landMask', True)
    params.put('useSRTM', True)
    params.put('shorelineExtension', JInteger(2))

    return GPF.createProduct('Land-Sea-Mask', params, product)

def subset_to_aoi(product, wkt_string):
    print("Subsetting to AOI...")

    HashMap = jpy.get_type('java.util.HashMap')
    params = HashMap()
    params.put('geoRegion', wkt_string)
    params.put('copyMetadata', True)

    return GPF.createProduct('Subset', params, product)

def reorder_bands_explicitly(product, output_band_order):
    print("Rebuilding product to enforce band order:", output_band_order)

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
        band_type = source_band.getDataType()

        new_band = BandType(band_name, band_type, width, height)
        new_product.addBand(new_band)

        # Read and assign raster data
        source_pixels = source_band.readPixels(0, 0, width, height, jpy.array('float', width * height))
        raster = ProductData.createInstance(ProductData.TYPE_FLOAT32, width * height)
        for i in range(len(source_pixels)):
            raster.setElemFloatAt(i, source_pixels[i])
        new_band.setRasterData(raster)

    return new_product


def write_product(product, output_path, format='GeoTIFF'):
    ProductIO.writeProduct(product, output_path, format)

def main():
    initialize_snap()

    input_path = r'/Volumes/External/TJ_estuary/01_data/sentinel_1/01_JunethroughDec/S1A_IW_GRDH_1SDV_20240602T134457_20240602T134522_054145_069598_F5E2.zip'

    # Derive filename
    input_filename = os.path.basename(input_path)
    basename = os.path.splitext(input_filename)[0]
    output_path = f"/Users/ereilly/Documents/code/autoSAR_preprocessing/test/output/{basename}_pre.tif"

    print("Loading product...")
    product = load_product(input_path)

    print("Calibrating...")
    calibrated = calibrate_product(product)
    ProductUtils.copyTiePointGrids(product, calibrated)

    print("Applying multilook...")
    multilooked = apply_multilook(calibrated)

    print("Adding incang band...")
    multilooked = add_incang_band(multilooked)

    print("Geocoding...")
    geocoded = ellipsoid_correction(multilooked)

    print("Applying land-sea mask...")
    masked = apply_land_sea_mask(geocoded)

    print("Subsetting to AOI...")
    wkt_aoi = "POLYGON ((-117.25708 32.314991, -117.04834 32.314991, -117.04834 32.655563, -117.25708 32.655563, -117.25708 32.314991))"
    subset = subset_to_aoi(masked, wkt_aoi)

    print("Reordering bands...")
    ordered = reorder_bands_explicitly(subset, ['Sigma0_VV', 'incang'])

    print("Writing GeoTIFF output...")
    write_product(ordered, output_path)

    print("Processing complete.")

if __name__ == '__main__':
    main()
