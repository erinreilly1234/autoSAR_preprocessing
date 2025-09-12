# ENVI to GeoTIFF Batch Converter

This Python script converts **ENVI `.dat/.hdr` files** into **GeoTIFFs** with georeferencing applied from a reference raster.

Replace path with your own reference file.

---

## Features

- Batch converts all `.dat/.hdr` files in a folder (including subfolders).
- Copies **projection and geotransform** from a user-specified reference raster.
- Reference can be **any georeferenced raster format supported by GDAL** (GeoTIFF, ENVI, IMG, etc.).
- Falls back to **WGS84 (EPSG:4326)** if no reference is available.
- Saves converted `.tif` files alongside the original `.dat/.hdr`.

---

## Requirements

Python 3 with the following packages:

```bash
pip install spectral gdal numpy
