# ----------------------------------------------------------------------------------------------------
# Import Required Libraries
# ----------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import os
from glob import glob
import pylandstats as pls
import warnings
import rasterio as rio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import rioxarray

# Suppress warnings to keep output clean
warnings.filterwarnings('ignore')


# ----------------------------------------------------------------------------------------------------
# Define File and Directory Paths
# ----------------------------------------------------------------------------------------------------
# Main directory containing raw raster and vector datasets
MAIN_DATA_DIR = '/beegfs/halder/DATA'

# GitHub-linked project directory where processed data and results are stored
PROJECT_DATA_DIR = '/beegfs/halder/GITHUB/Landscape-Analysis/data'

# Temporary directory used for storing intermediate files
TEMP_DIR = os.path.join(PROJECT_DATA_DIR, 'TEMP')


# ----------------------------------------------------------------------------------------------------
# Load Hexagonal Grid (5 km² resolution) for Germany
# ----------------------------------------------------------------------------------------------------
EPSG = 25832  # Use ETRS89 / UTM Zone 32N as the projection (suitable for Germany)

# Path to grid shapefile
GRID_PATH = os.path.join(PROJECT_DATA_DIR, 'VECTOR', f'DE_Hexbins_2.5sqkm_EPSG_{EPSG}.shp')

# Load grid as a GeoDataFrame and retain relevant columns
grids_gdf = gpd.read_file(GRID_PATH)
grids_gdf = grids_gdf[['id', 'geometry']]
grids_gdf['id'] = grids_gdf['id'].astype(int)

print('Successfully read the grids!')


# ----------------------------------------------------------------------------------------------------
# Compute Landscape Metrics Using PyLandStats
# ----------------------------------------------------------------------------------------------------
out_dir = os.path.join(PROJECT_DATA_DIR, 'OUTPUT', 'Landscape_Metrics_2.5KM')
os.makedirs(out_dir, exist_ok=True)

# Path to ESA WorldCover LULC raster (10 m resolution, 2021)
lulc_file_path = os.path.join(MAIN_DATA_DIR, 'ESA_WORLDCOVER_10M_2021_V200', 'ESA_WorldCover_2021_DE_WGS84.tif')
reprojected_raster_path = os.path.join(TEMP_DIR, f'ESA_WorldCover_2021_DE_EPSG_{EPSG}.tif')

# Target CRS (Coordinate Reference System)
dst_crs = f'EPSG:{EPSG}'

# Open source raster
with rio.open(lulc_file_path) as src:
    transform, width, height = calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *src.bounds)
    
    kwargs = src.meta.copy()
    kwargs.update({
        'crs': dst_crs,
        'transform': transform,
        'width': width,
        'height': height
    })

    # Write reprojected raster
    with rio.open(reprojected_raster_path, 'w', **kwargs) as dst:
        for i in range(1, src.count + 1):
            reproject(
                source=rio.band(src, i),
                destination=rio.band(dst, i),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest
            )
            
print('Raster Saved Successfully!')

# Create ZonalAnalysis object for computing landscape metrics per grid zone
za = pls.ZonalAnalysis(
    reprojected_raster_path,  # Use the reprojected raster
    zones=grids_gdf,
    zone_index='id',
    neighborhood_rule=8       # 8-neighbor connectivity for landscape pattern analysis
)

# Compute class-level metrics (per land cover class) for each zone
class_metrics_df = za.compute_class_metrics_df().reset_index()
class_metrics_df.to_csv(os.path.join(out_dir, 'class_metrics.csv'), index=False)

# Compute landscape-level metrics (overall structure) for each zone
landscape_metrics_df = za.compute_landscape_metrics_df().reset_index()
landscape_metrics_df.to_csv(os.path.join(out_dir, 'landscape_metrics.csv'), index=False)

print('Landscape metrics computation complete!')