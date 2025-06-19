from esa_snappy import ProductIO, GPF, HashMap, jpy, ProductUtils

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
    width = product.getSceneRasterWidth()
    height = product.getSceneRasterHeight()

    print("Available tie-point grids:")
    for i in range(product.getNumTiePointGrids()):
        print(" -", product.getTiePointGridAt(i).getName())

    tie_grid = product.getTiePointGrid('incident_angle')
    if tie_grid is None:
        raise RuntimeError("❌ 'incident_angle' tie-point grid not found in the product.")

    ProductData = jpy.get_type('org.esa.snap.core.datamodel.ProductData')
    Band = jpy.get_type('org.esa.snap.core.datamodel.Band')

    incang_band = Band('incang', ProductData.TYPE_FLOAT32, width, height)
    product.addBand(incang_band)

    incang_data = incang_band.createCompatibleRasterData()
    for y in range(height):
        for x in range(width):
            value = tie_grid.getPixelFloat(x, y)
            incang_data.setElemFloatAt(y * width + x, value)
    incang_band.setRasterData(incang_data)

# Optional: Disabled for now due to runtime issues
"""
def ellipsoid_correction(product, proj='WGS84(DD)'):
    print("\t🌍 Applying Ellipsoid Correction (Generic Geocoding)...")
    band_names = product.getBandNames()
    bands = ",".join(band_names)
    params = HashMap()
    params.put('sourceBands', bands)
    params.put('imgResamplingMethod', 'BILINEAR_INTERPOLATION')
    params.put('mapProjection', proj)
    return GPF.createProduct('Ellipsoid-Correction-GG', params, product)
"""

def write_product(product, output_path):
    ProductIO.writeProduct(product, output_path, 'BEAM-DIMAP')

def main():
    initialize_snap()

    input_path = r'/Volumes/External/TJ_estuary/01_data/sentinel_1/01_JunethroughDec/S1A_IW_GRDH_1SDV_20240602T134457_20240602T134522_054145_069598_F5E2.zip'
    output_path = '/Users/ereilly/Documents/code/autoSAR_preprocessing/test/output/cal_ml_incang_no_geocoding.dim'

    print("📥 Loading product...")
    product = load_product(input_path)

    print("⚙️  Calibrating...")
    calibrated = calibrate_product(product)
    ProductUtils.copyTiePointGrids(product, calibrated)

    print("🌀 Applying multilook...")
    multilooked = apply_multilook(calibrated)

    print("📡 Adding incident angle band...")
    add_incang_band(multilooked)

    # Temporarily skip geocoding step
    print("⚠️  Ellipsoid correction is currently disabled. Output is not geocoded.")
    geocoded = multilooked  # Bypass geocoding for now

    print("💾 Writing output...")
    write_product(geocoded, output_path)

    print("✅ Processing complete. Output saved at:")
    print(output_path)

if __name__ == '__main__':
    main()