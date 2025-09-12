## DOESN"T WORK :((((##3

import os
import csv
from esa_snappy import ProductIO, GPF, HashMap, jpy
import fiona
from shapely.geometry import shape, Point

GeoPos = jpy.get_type('org.esa.snap.core.datamodel.GeoPos')
PixelPos = jpy.get_type('org.esa.snap.core.datamodel.PixelPos')


INPUT_FOLDER       = '/Volumes/External/TJ_SAR/01_data/03_2022_2025_gooddays'
SHAPE_FILE         = '/Volumes/External/TJ_SAR/01_data/shapefiles/Outflow.shp'
OUTPUT_DIR         = '/Volumes/External/TJ_SAR/analysis/windFields'
OUTPUT_CSV         = '/Volumes/External/TJ_SAR/analysis/wind_SAR.csv'

# Processing parameters
MASK_LAYER         = 'Land-Sea-Mask'       # e.g. 'LandSeaMask'
OUTPUT_SIGMA_BAND  = True                # True to output Sigma0 band
WIND_ALGORITHM     = 'CMOD5'             # 'CMOD4', 'CMOD5', etc.
CONTRIBUTING_BAND  = 'Sigma0_VV'         # e.g. 'Sigma0_VV' or 'Sigma0_VH'
# ====================

def process_product(file_path, targets, csv_writer):
    # 1. Read product
    product = ProductIO.readProduct(file_path)

    # 2. Land-Sea Mask
    ls_mask = GPF.createProduct(MASK_LAYER, HashMap(), product)

    # 3. Calibration to Sigma0
    calib_params = HashMap()
    calib_params.put('outputSigmaBand', OUTPUT_SIGMA_BAND)
    calib = GPF.createProduct('Calibration', calib_params, ls_mask)


    # 4. Wind Field Retrieval (correct operator alias)
    wf_params = HashMap()
    wf_params.put('windAlgorithm', WIND_ALGORITHM)
    wf_params.put('contributingBand', CONTRIBUTING_BAND)
    wind_prod = GPF.createProduct('Wind-Field-Estimation', wf_params, calib)

    # Debug: list available band names
    bands = [b.getName() for b in wind_prod.getBands()]
    print(f"Bands in wind product: {bands}")

    # Determine correct band names for speed and direction
    speed_names = ['wind_speed', 'windSpeed', 'wind speed']
    dir_names = ['wind_dir', 'windDir', 'wind direction']
    band_speed = next((wind_prod.getBand(n) for n in speed_names if wind_prod.getBand(n)), None)
    band_dir   = next((wind_prod.getBand(n) for n in dir_names if wind_prod.getBand(n)), None)
    if not band_speed or not band_dir:
        raise RuntimeError(f"Wind bands not found. Available: {bands}")

    # 5. Retrieve geocoding
    geo = wind_prod.getSceneGeoCoding()

    # 6. Sample each target at its centroid
    for feat in targets:
        geom = shape(feat['geometry'])
        lon, lat = geom.centroid.x, geom.centroid.y

        gp = GeoPos(lat, lon)
        pp = PixelPos()
        geo.getPixelPos(gp, pp)
        px, py = int(round(pp.x)), int(round(pp.y))

        speed = band_speed.readPixels(px, py, 1, 1, [0.0])[0]
        direction = band_dir.readPixels(px, py, 1, 1, [0.0])[0]

        csv_writer.writerow({
            'product': os.path.basename(file_path),
            'feature_id': feat.get('id') or feat['properties'].get('id', ''),
            'lon': lon,
            'lat': lat,
            'wind_speed': speed,
            'wind_dir': direction
        })


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load target features once
    with fiona.open(SHAPE_FILE, 'r') as src:
        targets = list(src)

    # Prepare CSV
    with open(OUTPUT_CSV, 'w', newline='') as csvfile:
        fieldnames = ['product', 'feature_id', 'lon', 'lat', 'wind_speed', 'wind_dir']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Process each product
        for root, _, files in os.walk(INPUT_FOLDER):
            for fname in files:
                if fname.lower().endswith(('.dim', '.tif', '.zip')):
                    path = os.path.join(root, fname)
                    print(f"Processing {path}...")
                    process_product(path, targets, writer)

    print(f"Processing complete. Results written to {OUTPUT_CSV}")

if __name__ == '__main__':
    main()
