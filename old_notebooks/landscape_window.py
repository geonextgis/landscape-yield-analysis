import os
import numpy as np
import rasterio
from rasterio.mask import mask
from skimage.util import view_as_windows
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import mapping


def shannon_diversity(window: np.ndarray) -> float:
    """Calculate the Shannon Diversity Index (SHDI) for a given 2D window."""
    values, counts = np.unique(window, return_counts=True)
    proportions = counts / counts.sum()
    return -np.sum(proportions * np.log(proportions))


def _process_row(args):
    """Helper for processing a row of windows in parallel."""
    row_idx, windows_row = args
    return [shannon_diversity(windows_row[j]) for j in range(windows_row.shape[0])]


def calculate_shdi_for_grids(grids_gdf: gpd.GeoDataFrame, input_raster_path: str, output_dir: str, window_size: int = 3, n_jobs: int = None):
    """
    Compute SHDI for each grid cell in a GeoDataFrame and save results to individual files.

    Parameters
    ----------
    grids_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing grid geometries.
    input_raster_path : str
        Path to the input LULC raster.
    output_dir : str
        Directory to save SHDI rasters for each grid.
    window_size : int
        Size of the moving window (must be odd).
    n_jobs : int
        Number of parallel jobs.
    """
    os.makedirs(output_dir, exist_ok=True)

    with rasterio.open(input_raster_path) as src:
        for idx, row in tqdm(grids_gdf.iterrows(), total=len(grids_gdf), desc="Processing Grids"):
            geom = [mapping(row.geometry)]

            try:
                out_image, out_transform = mask(src, geom, crop=True)
                out_image = out_image[0]  # Extract band 1
            except ValueError:
                print(f"Skipping grid {idx} - no overlap with raster.")
                continue

            # Replace NaNs
            out_image = np.where(np.isnan(out_image), 0, out_image)

            pad = window_size // 2
            padded_data = np.pad(out_image, pad, mode='edge')

            windows = view_as_windows(padded_data, (window_size, window_size))
            rows, cols = windows.shape[:2]

            tasks = [(i, windows[i]) for i in range(rows)]
            sdi_array = np.zeros((rows, cols), dtype=np.float32)

            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                results = list(executor.map(_process_row, tasks))

            for i, row_values in enumerate(results):
                sdi_array[i, :] = row_values

            # Define output profile
            out_profile = src.profile.copy()
            out_profile.update({
                'height': rows,
                'width': cols,
                'transform': rasterio.transform.from_origin(
                    out_transform.c + pad * src.res[0],
                    out_transform.f - pad * src.res[1],
                    src.res[0],
                    src.res[1]
                ),
                'dtype': rasterio.float32,
                'count': 1,
                'compress': 'lzw',
                'nodata': np.nan
            })

            output_path = os.path.join(output_dir, f"shdi_grid_{idx}.tif")
            with rasterio.open(output_path, 'w', **out_profile) as dst:
                dst.write(sdi_array, 1)