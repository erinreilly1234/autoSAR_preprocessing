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
    print(" Creating and merging 'incang' band from 'incident_angle'...")

    tie_grids = [product.getTiePointGridAt(i).getName() for i in range(product.getNumTiePointGrids())]
    if 'incident_angle' not in tie_grids:
        raise RuntimeError("**** 'incident_angle' tie-point grid not found.****")

    # Step 1: Build a new product with only the incang band
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
    params.put('retainExistingBands', False)  # Only generate incang
    incang_product = GPF.createProduct('BandMaths', params, product)

    # Step 2: Merge incang into original product
    merge_params = HashMap()
    merge_params.put('sourceProductNames', 'master,slave')
    merge_params.put('resamplingMethod', 'NEAREST_NEIGHBOUR')
    merge_params.put('geodeticTiePoints', True)

    sources = HashMap()
    sources.put('master', product)
    sources.put('slave', incang_product)

    merged = GPF.createProduct('BandMerge', merge_params, sources)

    print("-- 'incang' band merged with original product.")
    return merged

def ellipsoid_correction(product, proj='WGS84(DD)'):
    print("\t Applying Ellipsoid Correction (Generic Geocoding)...")
    band_names = list(product.getBandNames())  # Convert Java array to Python list
    print("--- Bands to geocode:", band_names)

    params = HashMap()
    params.put('sourceBands', ",".join(band_names))
    params.put('imgResamplingMethod', 'BILINEAR_INTERPOLATION')
    params.put('mapProjection', proj)

    return GPF.createProduct('Ellipsoid-Correction-GG', params, product)

def apply_land_sea_mask(product):
    print(" Applying Land-Sea Mask...")

    HashMap = jpy.get_type('java.util.HashMap')
    params = HashMap()
    params.put('landMask', True)
    params.put('useSRTM', True)
    JInteger = jpy.get_type('java.lang.Integer')
    params.put('shorelineExtension', JInteger(2))
    # Optional: use 'sourceBands' or 'geometry' if needed

    return GPF.createProduct('Land-Sea-Mask', params, product)

def subset_to_aoi(product, wkt_string):
    print(" Subsetting to AOI using WKT...")

    HashMap = jpy.get_type('java.util.HashMap')
    params = HashMap()
    params.put('geoRegion', wkt_string)
    params.put('copyMetadata', True)
    return GPF.createProduct('Subset', params, product)



def write_product(product, output_path, format='BEAM-DIMAP'):
    ProductIO.writeProduct(product, output_path, format)

def main():
    initialize_snap()

    # Paths
    input_path = r'/Volumes/External/TJ_estuary/01_data/sentinel_1/01_JunethroughDec/S1A_IW_GRDH_1SDV_20240602T134457_20240602T134522_054145_069598_F5E2.zip'
    temp_path = '/Users/ereilly/Documents/code/autoSAR_preprocessing/test/output/intermediate_with_incang'
    final_path = '/Users/ereilly/Documents/code/autoSAR_preprocessing/test/output/final_geocoded_output'
    wkt_aoi = "POLYGON ((-117.339478 32.328917, -117.051086 32.328917, -117.051086 32.644, -117.339478 32.644, -117.339478 32.328917))"

    # === STAGE 1: Preprocessing and save ===
    print("- Loading product...")
    product = load_product(input_path)

    print("️-  Calibrating...")
    calibrated = calibrate_product(product)
    ProductUtils.copyTiePointGrids(product, calibrated)

    print("- Applying multilook...")
    multilooked = apply_multilook(calibrated)

    print("- Adding 'incang' band...")
    multilooked = add_incang_band(multilooked)

    print(f"- Saving intermediate product to {temp_path}.dim ...")
    write_product(multilooked, temp_path)

    # === STAGE 2: Reload and geocode ===
    print("- Reloading product with incang band...")
    reloaded = load_product(temp_path + '.dim')

    print("-  Geocoding using Ellipsoid-Correction-GG...")
    geocoded = ellipsoid_correction(reloaded)

    print("- Masking land areas using Land-Sea Mask...")
    masked = apply_land_sea_mask(geocoded)

    print("- Subsetting to area of interest (AOI)...")
    subset = subset_to_aoi(masked, wkt_aoi)

    print(f"- Writing final clipped output to {final_path}_masked_subset.dim ...")
    write_product(subset, final_path + '_masked_subset')

    print("----- Done! Have a great day! ------")

if __name__ == '__main__':
    main()
