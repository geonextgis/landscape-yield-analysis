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
import rasterio.mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterstats import zonal_stats
import argparse
from tqdm import tqdm

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
# Load Hexagonal Grid for Germany
# ----------------------------------------------------------------------------------------------------
# argparse setup
parser = argparse.ArgumentParser(description="Process SQR and SWF for a given hexagonal grid file.")
parser.add_argument('--distance', type=str, required=True, help="Distance of the hexagonal grid file (e.g., 2.5, 5, 10)")
args = parser.parse_args()

DISTANCE = args.distance
EPSG = 25832  # Use ETRS89 / UTM Zone 32N as the projection (suitable for Germany)

# Path to grid shapefile
GRID_PATH = os.path.join(PROJECT_DATA_DIR, 'VECTOR', f'DE_Hexbins_{DISTANCE}sqkm_EPSG_{EPSG}.shp')

# Load grid as a GeoDataFrame and retain relevant columns
grids_gdf = gpd.read_file(GRID_PATH)
grids_gdf = grids_gdf[['id', 'geometry']]
grids_gdf['id'] = grids_gdf['id'].astype(int)

print('Grids Shape:', grids_gdf.shape)
print('Successfully read the grids!')


# ----------------------------------------------------------------------------------------------------
# Compute SQR using Rasterstats
# ----------------------------------------------------------------------------------------------------
# out_dir = os.path.join(PROJECT_DATA_DIR, 'OUTPUT', f'Landscape_Metrics_{DISTANCE}KM')
# os.makedirs(out_dir, exist_ok=True)

# # Path to soil quality raster (250 m resolution)
# soil_file_path = os.path.join(MAIN_DATA_DIR, 'DE_Soil_Quality_Rating_250m', 'sqr1000_250_v10.tif')
# reprojected_raster_path = os.path.join(TEMP_DIR, f'sqr1000_250_v10_{DISTANCE}.tif')

# # Target CRS (Coordinate Reference System)
# dst_crs = f'EPSG:{EPSG}'

# # Open source raster
# with rio.open(soil_file_path) as src:
#     transform, width, height = calculate_default_transform(
#         src.crs, dst_crs, src.width, src.height, *src.bounds)
    
#     kwargs = src.meta.copy()
#     kwargs.update({
#         'crs': dst_crs,
#         'transform': transform,
#         'width': width,
#         'height': height
#     })

#     # Write reprojected raster
#     with rio.open(reprojected_raster_path, 'w', **kwargs) as dst:
#         for i in range(1, src.count + 1):
#             reproject(
#                 source=rio.band(src, i),
#                 destination=rio.band(dst, i),
#                 src_transform=src.transform,
#                 src_crs=src.crs,
#                 dst_transform=transform,
#                 dst_crs=dst_crs,
#                 resampling=Resampling.nearest
#             )
            
# print('Raster Saved Successfully!')

# # Collect valid zone indices
# valid_indices = []

# with rio.open(reprojected_raster_path) as src:
#     for idx, row in tqdm(grids_gdf.iterrows(), total=len(grids_gdf)):
#         try:
#             mask, transform = rio.mask.mask(src, [row['geometry']], crop=True)
#             unique_vals = np.unique(mask)
#             if len(unique_vals) <= 1:  # Only background or no-data
#                 print(f"Zone {row['id']} skipped: {unique_vals}")
#                 continue
#             valid_indices.append(idx)
#         except Exception as e:
#             print(f"Zone {row['id']} caused error: {e}")
#             continue

# # Filter the GeoDataFrame
# grids_gdf = grids_gdf.loc[valid_indices].copy()

# # Calculate zonal stats
# stats = zonal_stats(grids_gdf, reprojected_raster_path, stats=["mean", "std"])

# # Add results to GeoDataFrame
# zones_stats = grids_gdf.copy()
# for key in stats[0].keys():
#     zones_stats[key] = [s[key] for s in stats]

# zones_stats.rename(columns={"mean": 'SQR_MEAN', 'std': 'SQR_STD'}, inplace=True)
# zones_stats = zones_stats[['id', 'SQR_MEAN', 'SQR_STD']]

# # Save the data
# zones_stats.to_csv(os.path.join(out_dir, f'soil_quality_rating_(sqr).csv'), index=False)
# print('Soil Quality Rating computation complete!')


# ----------------------------------------------------------------------------------------------------
# Compute Percentage of Small Woody Features using Rasterstats
# ----------------------------------------------------------------------------------------------------
out_dir = os.path.join(PROJECT_DATA_DIR, 'OUTPUT', f'Landscape_Metrics_{DISTANCE}KM')
os.makedirs(out_dir, exist_ok=True)

# Path to small woody feature raster (5 m resolution)
swf_file_path = os.path.join(MAIN_DATA_DIR, 'DE_Small_Woody_Features_5m', 'HRL_Small_Woody_Features_2018_005m.tif')
reprojected_raster_path = os.path.join(TEMP_DIR, f'HRL_Small_Woody_Features_2018_005m_{DISTANCE}.tif')

# Target CRS (Coordinate Reference System)
dst_crs = f'EPSG:{EPSG}'

# Open source raster
with rio.open(swf_file_path) as src:
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

# Collect valid zone indices
valid_indices = []

with rio.open(reprojected_raster_path) as src:
    for idx, row in tqdm(grids_gdf.iterrows(), total=len(grids_gdf)):
        try:
            mask, transform = rio.mask.mask(src, [row['geometry']], crop=True)
            unique_vals = np.unique(mask)
            if len(unique_vals) <= 1:  # Only background or no-data
                print(f"Zone {row['id']} skipped: {unique_vals}")
                continue
            valid_indices.append(idx)
        except Exception as e:
            print(f"Zone {row['id']} caused error: {e}")
            continue

# Filter the GeoDataFrame
grids_gdf = grids_gdf.loc[valid_indices].copy()

# Calculate zonal stats
stats = zonal_stats(grids_gdf, reprojected_raster_path, stats=["sum"])

# Calculate the Grid area
grids_gdf['area_ha'] = grids_gdf.geometry.area / 10000

# Add results to GeoDataFrame
zones_stats = grids_gdf.copy()
for key in stats[0].keys():
    zones_stats[key] = [s[key] for s in stats]

zones_stats.rename(columns={"sum": 'SWF_area'}, inplace=True)
zones_stats['SWF_area_ha'] = (zones_stats['SWF_area'] * (5**2)) / 10000
zones_stats = zones_stats[['id', 'geometry', 'area_ha', 'SWF_area_ha']]
zones_stats['SWF_area_perc'] = (zones_stats['SWF_area_ha'] / zones_stats['area_ha']) * 100
zones_stats = zones_stats[['id', 'area_ha', 'SWF_area_ha', 'SWF_area_perc']]

# Save the data
zones_stats.to_csv(os.path.join(out_dir, f'small_woody_featuers_percentage.csv'), index=False)
print('Small Woody Features Percentage computation complete!')



































