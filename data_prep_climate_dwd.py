# ----------------------------------------------------------------------------------------------------
# Parse Required Arguments
# ----------------------------------------------------------------------------------------------------
import argparse

parser = argparse.ArgumentParser(description="Prepare climate data from DWD with custom distance")
parser.add_argument("--distance", type=str, default=10, help="Distance in kilometers for station selection or filtering")
args = parser.parse_args()

DISTANCE = args.distance

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
import warnings
import argparse
from tqdm import tqdm
import json
from shapely.geometry import Point
from datetime import datetime, timedelta
from scipy.stats import weibull_min
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor

# Suppress warnings to keep output clean
warnings.filterwarnings('ignore')


# ----------------------------------------------------------------------------------------------------
# Define File and Directory Paths
# ----------------------------------------------------------------------------------------------------
# Main directory containing raw raster and vector datasets
MAIN_DATA_DIR = '/beegfs/halder/DATA'

# GitHub-linked project directory where processed data and results are stored
PROJECT_DATA_DIR = '/beegfs/halder/GITHUB/Landscape-Analysis/data'

# Define the DWD data directory
DWD_DATA_DIR = '/beegfs/common/data/climate/dwd/csvs/germany_ubn_1951-01-01_to_2024-08-30'

# Temporary directory used for storing intermediate files
TEMP_DIR = os.path.join(PROJECT_DATA_DIR, 'TEMP')

# ----------------------------------------------------------------------------------------------------
# Load Hexagonal Grid and other shapefiles for Germany
# ----------------------------------------------------------------------------------------------------
EPSG = 25832  # Use ETRS89 / UTM Zone 32N as the projection (suitable for Germany)

# Path to grid shapefile
GRID_PATH = os.path.join(PROJECT_DATA_DIR, 'VECTOR', f'DE_Hexbins_{DISTANCE}sqkm_EPSG_{EPSG}.shp')

# Define the out directory
OUT_DIR = os.path.join(PROJECT_DATA_DIR, 'OUTPUT', f'Landscape_Metrics_{DISTANCE}KM', 'Climate')

# Load grid as a GeoDataFrame and retain relevant columns
grids_gdf = gpd.read_file(GRID_PATH)
grids_gdf = grids_gdf[['id', 'geometry']]
grids_gdf['id'] = grids_gdf['id'].astype(int)

print('Successfully read the grids!')

# Read the Germany shapefile
DE_gdf = gpd.read_file(os.path.join(MAIN_DATA_DIR, 'DE_NUTS', 'DE_NUTS_3.shp'))
DE_gdf = DE_gdf[DE_gdf['LEVL_CODE']==1] 
DE_gdf = DE_gdf.to_crs(f'EPSG:{EPSG}')

grids_centroids = grids_gdf.copy()
grids_centroids['geometry'] = grids_gdf.centroid

grids_centroids = gpd.sjoin_nearest(left_df=grids_centroids, right_df=DE_gdf[['NUTS_NAME', 'geometry']],
                                    how='inner')

grids_gdf = pd.merge(left=grids_gdf, right=grids_centroids[['id', 'NUTS_NAME']], how='inner', on='id')

# Specify the DE DWD grip file path
DE_DWD_json_path = os.path.join(MAIN_DATA_DIR, 'DE_DWD_Lat_Lon', 'latlon_to_rowcol.json')

with open(DE_DWD_json_path) as f:
    data = json.load(f)

# Convert data to GeoDataFrame
records = []
for coord, index in data:
    lat, lon = coord
    row, col = index
    point = Point(lon, lat)
    records.append({'row': row, 'col': col, 'geometry': point})

latlon_gdf = gpd.GeoDataFrame(records, geometry='geometry', crs='EPSG:4326')
latlon_gdf = latlon_gdf.to_crs(f'EPSG:{EPSG}')

# Apply spatial join
grids_gdf_dwd = gpd.sjoin(left_df=grids_gdf, right_df=latlon_gdf, how='inner', predicate='intersects')
grids_gdf_dwd.drop(columns=['index_right', 'geometry'], inplace=True)

for col in ['id', 'row', 'col']:
    grids_gdf_dwd[col] = grids_gdf_dwd[col].astype(int)

grids_gdf_dwd['rowcol'] = list(zip(grids_gdf_dwd['row'], grids_gdf_dwd['col']))

# ----------------------------------------------------------------------------------------------------
# Process the phenolgy data
# ----------------------------------------------------------------------------------------------------
# Define the crop name
CROP = 'WW'

# Read the phenology data
phenology_df = pd.read_csv(os.path.join(MAIN_DATA_DIR, 'DE_Crop_Phenology', 'WW_phenology_1999_2021.csv'))
phenology_df.rename(columns={'Sowing_DOY': 'Sowing_DATE', 'Flowering_DOY': 'Flowering_DATE', 'Harvest_DOY': 'Harvest_DATE'}, inplace=True)
for date_col in ['Sowing_DATE', 'Flowering_DATE', 'Harvest_DATE']:
    event = date_col.split("_")[0]
    phenology_df[date_col] = pd.to_datetime(phenology_df[date_col], format='%Y-%m-%d')
    phenology_df[f'{event}_DOY'] = phenology_df[date_col].dt.dayofyear

# Group by STATE_ID and take the median of DOYs
median_doys = phenology_df.groupby('STATE_NAME')[['Sowing_DOY', 'Harvest_DOY']].median().round().astype(int).reset_index()
phenology_df = phenology_df.merge(median_doys, on='STATE_NAME', suffixes=('', '_MEDIAN'))

# Create full year range
years = list(range(1952, 2024))

# Create cartesian product of states and years
phenology_median = pd.MultiIndex.from_product([median_doys['STATE_NAME'], years], names=['STATE_NAME', 'Year']).to_frame(index=False)

# Merge back state names and median DOYs
phenology_median = phenology_median.merge(median_doys, on='STATE_NAME', how='left')

def doy_to_date(year, doy, event='sow'):
    if event == 'sow':
        return datetime(year-1, 1, 1) + timedelta(days=int(doy) - 1)
    else:
        return datetime(year, 1, 1) + timedelta(days=int(doy) - 1)

phenology_median['Sowing_DATE'] = phenology_median.apply(lambda row: doy_to_date(row['Year'], row['Sowing_DOY'], event='sow'), axis=1)
phenology_median['Harvest_DATE'] = phenology_median.apply(lambda row: doy_to_date(row['Year'], row['Harvest_DOY'], event='harvest'), axis=1)


# ----------------------------------------------------------------------------------------------------
# Process the climate data
# ----------------------------------------------------------------------------------------------------
def get_phenology(state_name, year):
    index = phenology_median[(phenology_median['STATE_NAME']==state_name) & (phenology_median['Year']==year)].index[0]
    data_dict = phenology_median.loc[index].to_dict()
    return data_dict

def compute_climate_indices(hex_id, row_cols_id, phenology, state_name, clim_df):
    """
    Compute climate indices per year based on phenology windows.

    Parameters:
        hex_id (str): Unique grid or hexagon ID.
        row_cols_id (str): Row-col or spatial index identifier.
        phenology (pd.DataFrame): Contains 'STATE_NAME', 'Year', 'Sowing_DATE', 'Harvest_DATE'.
        state_name (str): German federal state name (e.g., 'Bremen').
        clim_df (pd.DataFrame): Daily weather data with 'Date', 'Precipitation', 'TempMin', 'TempMax', 'Radiation'.

    Returns:
        pd.DataFrame: Yearly climate indices for the given hex_id and state.
    """

    def get_phenology(state, year):
        try:
            row = phenology[
                (phenology['STATE_NAME'] == state) &
                (phenology['Year'] == year)
            ].iloc[0]
            return {
                'Sowing_DATE': row['Sowing_DATE'],
                'Harvest_DATE': row['Harvest_DATE']
            }
        except IndexError:
            return None

    final_data = []

    for year in phenology['Year'].unique():
        pheno = get_phenology(state_name, year)
        if pheno is None:
            continue

        start_date = pheno['Sowing_DATE']
        end_date = pheno['Harvest_DATE']

        clim_df_masked = clim_df[
            (clim_df["Date"] >= start_date) &
            (clim_df["Date"] <= end_date)
        ]

        # Precipitation
        prcp = clim_df_masked['Precipitation']
        prcp_max = prcp.max()
        prcp_sum = prcp.sum()
        positive_precip = prcp[prcp > 0.1]

        if len(positive_precip) > 5:
            shape, _, _ = weibull_min.fit(positive_precip, floc=0)
            prcp_var = shape
        else:
            prcp_var = None

        # Temperature Min
        tmin = clim_df_masked['TempMin']
        tmin_min = tmin.min()
        tmin_mean = tmin.mean()
        tmin_std = tmin.std()

        # Temperature Max
        tmax = clim_df_masked['TempMax']
        tmax_max = tmax.max()
        tmax_mean = tmax.mean()
        tmax_std = tmax.std()

        # Radiation
        rad = clim_df_masked['Radiation'] / 1000
        rad_min = rad.min()
        rad_max = rad.max()
        rad_mean = rad.mean()
        rad_std = rad.std()

        year_dict = {
            'id': hex_id,
            'rowcol': row_cols_id,
            'year': year,
            'Prcp_Max': prcp_max,
            'Prcp_Sum': prcp_sum,
            'Prcp_Var_Index': prcp_var,
            'Tmin_Min': tmin_min,
            'Tmin_Mean': tmin_mean,
            'Tmin_Std': tmin_std,
            'Tmax_Max': tmax_max,
            'Tmax_Mean': tmax_mean,
            'Tmax_Std': tmax_std,
            'Rad_Min': rad_min,
            'Rad_Max': rad_max,
            'Rad_Mean': rad_mean,
            'Rad_Std': rad_std
        }

        final_data.append(year_dict)

    final_data = pd.DataFrame(final_data)
    final_data = final_data.astype({col: 'float32' for col in final_data.select_dtypes(include='float64').columns})
    return final_data

def extract_climate_data_by_grids(hex_id, out_dir):
    state_name = grids_gdf[grids_gdf['id']==hex_id]['NUTS_NAME'].iloc[0]
    row_cols = np.array(grids_gdf_dwd[grids_gdf_dwd['id']==hex_id]['rowcol'])
    file_paths = [os.path.join(DWD_DATA_DIR, str(row), f'daily_mean_RES1_C{col}R{row}.csv.gz') for row, col in row_cols]

    if os.path.exists(out_dir):
        print('Directory already exists!')

    else:
        os.makedirs(out_dir, exist_ok=True)
        print('Directory successfully created!')
        
    for f in file_paths:
        row_cols_id = os.path.basename(f).split("_")[-1].split(".")[0]
        try:
            clim_df = pd.read_csv(f, delimiter='\t')
            clim_df = clim_df[['Date', 'Precipitation', 'TempMin', 'TempMax', 'Radiation']]
            clim_df['Date'] = pd.to_datetime(clim_df['Date'])
    
            processed_df = compute_climate_indices(hex_id, row_cols_id, phenology_median, state_name, clim_df)
            out_data_name = f'{hex_id}_{row_cols_id}.csv'
            out_path = os.path.join(out_dir, out_data_name)
            processed_df.to_csv(out_path, index=False)
            print(f'ID: {hex_id} | DWD RowCol ID: {row_cols_id} | Status: Data saved at path: {out_path}')

        except:
            continue


# ----------------------------------------------------------------------------------------------------
# Run the extaction in parallel
# ----------------------------------------------------------------------------------------------------
# Prepare all hex_ids to be processed
hex_ids = grids_gdf_dwd['id'].unique()

# Use a function wrapper that takes just one argument (hex_id)
def wrapper(hex_id):
    extract_climate_data_by_grids(hex_id, OUT_DIR)


# ----------------------------------------------------------------------------------------------------
# Postprocess the data
# ----------------------------------------------------------------------------------------------------
def process_hex_data(hex_id):
    file_paths = [os.path.join(OUT_DIR, name) for name in file_names if name.startswith(f'{hex_id}_')]
    hex_dfs = [pd.read_csv(fp) for fp in file_paths]
    hex_df = pd.concat(hex_dfs, axis=0, ignore_index=True)

    # Groupby based on 'id' and 'year'
    hex_df_grouped = hex_df.groupby(by=['id', 'year']).mean(numeric_only=True).reset_index()
    return hex_df_grouped


# ----------------------------------------------------------------------------------------------------
# Run the script
# ----------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    # Run this section if you want to extract the climate data
    # with concurrent.futures.ProcessPoolExecutor(max_workers=70) as executor:
    #     executor.map(wrapper, hex_ids)
    # print('Climate data computation complete!')

    
    # Run this section if you want to post-process the climate data
    # Store the output file paths
    FINAL_OUT_DIR = os.path.join(PROJECT_DATA_DIR, 'OUTPUT', f'Landscape_Metrics_{DISTANCE}KM')
    
    out_file_paths = glob(os.path.join(OUT_DIR, '*.csv'))
    file_names = [os.path.basename(p) for p in out_file_paths]
    hex_ids = list(set([name.split("_")[0] for name in file_names]))

    # Run in parallel
    final_df = pd.DataFrame()
    
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_hex_data, hex_ids), total=len(hex_ids)))
    
    # Combine all results
    final_df = pd.concat(results, axis=0, ignore_index=True)
    final_df.to_csv(os.path.join(FINAL_OUT_DIR, f'{CROP.lower()}_climate.csv'), index=False)
    print('Climate data post-processing complete!')

