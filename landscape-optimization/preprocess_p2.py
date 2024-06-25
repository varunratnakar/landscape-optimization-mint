import pandas as pd
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from glob import glob
import os
from pathlib import Path
import rasterio
import matplotlib.pyplot as plt
from collections import defaultdict
import geopandas as gpd
from shapely.geometry import MultiPolygon
import time
import yaml

import dask.dataframe as dd
from dask.dataframe import from_pandas, concat
from dask.distributed import Client, LocalCluster

import math
from rasterio.merge import merge
import shutil
from rasterio.windows import Window
from rasterio.windows import from_bounds
from rasterio.transform import Affine
from rasterio.plot import show
from shapely.geometry import box
from time import time_ns
import heapq
import sys

import rioxarray as rxr
import xarray as xr

def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text

def get_truncated_ignitions(full_ignitions_df, burn_file_names):
    #TODO :FILTER OUT THE INTERSECTING RASTERS
    # raster_names = list(map(lambda x: os.path(x).stem, burn_file_names))
    raster_names = list(map(lambda x: os.path.splitext(os.path.basename(x))[0], burn_file_names))
    
    raster_names = list(map(lambda x: remove_prefix(x, 'burned_area-'), raster_names))

    full_ignition_idx = full_ignitions_df['filename'].isin(raster_names)
    truncated_ignitions_df = full_ignitions_df[full_ignition_idx]
    truncated_ignitions_df.reset_index(drop = True, inplace = True)

    return truncated_ignitions_df

## Call to get truncated ignitions list
#truncated_ignitions_df = get_truncated_ignitions(full_ignitions_df, burn_file_names)

def get_burn_area_values(original_values_df_path, hazard_raster_path):
    # read from original values df and create a new column for hazards
    # return the new values df

    values_df = pd.read_csv(original_values_df_path)

    hazard_raster = rasterio.open(hazard_raster_path)

    hazard_values = []
    intensity_values = []

    for name in values_df.filename:
        try:
            file_name = os.path.join(intensity_dir,'flamelen-{}.tif'.format(name))
            cur_raster = rasterio.open(file_name)
        except:
            hazard_values.append(0)
            intensity_values.append(0)
            continue

        xmin, ymin, xmax, ymax = cur_raster.bounds

        window = from_bounds(xmin, ymin, xmax, ymax, transform=hazard_raster.transform)
        local_hazard_values = hazard_raster.read(1, window=window)

        img = cur_raster.read(1)
        hazard_val = (local_hazard_values[np.where(img > 0)]).sum()
        hazard_values.append(hazard_val)
        intensity_val = img.sum()
        intensity_values.append(intensity_val)
        
    hazard_values = np.array(hazard_values)
    intensity_values = np.array(intensity_values)

    # add hazard values and intensity values to the values_df
    values_df['hazard'] = hazard_values
    values_df['intensity'] = intensity_values
    
    values_df['bldg_dmg'] = values_df['bldg_dmg'].astype(float) / total_files
    values_df['habitat_dmg'] = values_df['habitat_dmg'].astype(float) / total_files
    values_df['burn_area'] = values_df['burn_area'].astype(float) / total_files
    values_df['hazard'] = values_df['hazard'].astype(float) / total_files
    values_df['intensity'] = values_df['intensity'].astype(float) / total_files

    return values_df

def write_csv_to_file(file_path, data):
    data.to_csv(file_path, index=False)


def point_in_poly(point, polygon):
    return polygon.contains(Point(point))

def generate_prevention_df(rx_burn_units_path, values_df):
    prevention_df = pd.DataFrame(columns=['geometry','area', 'burned_area', 'bldg', 'habitat', 'intensity', 'hazard', 'covered_raster_ids'])

    func = lambda x, y : point_in_poly(x, y)
    vector_func = np.vectorize(func)

    f0s = []
    f1s = []
    f2s = []
    f3s = []
    f4s = []
    f5s = []
    covered_ids = []

    ignition_points = [(x, y) for x, y in zip(values_df.x_ignition, values_df.y_ignition)]
    ignition_points = np.array(ignition_points)
    
    rx_burn_units = gpd.read_file(rx_burn_units_path)
    rx_burn_units = rx_burn_units.to_crs('epsg:32610')

    burn_candidates = rx_burn_units.geometry
    contained_idx = None
    for poly in burn_candidates:
        contained_idx = list(map(lambda x: point_in_poly(x, poly), ignition_points))
        covered = np.where(contained_idx)[0]

        f1 = np.sum(values_df[contained_idx].burn_area)
        f2 = np.sum(values_df[contained_idx].bldg_dmg)
        f3 = np.sum(values_df[contained_idx].habitat_dmg)
        f4 = np.sum(values_df[contained_idx].hazard)
        f5 = np.sum(values_df[contained_idx].intensity)
        f0 = poly.area / (10**6)
        
        covered_ids.append(list(covered))
        f1s.append(f1)
        f2s.append(f2)
        f3s.append(f3)
        f4s.append(f4)
        f5s.append(f5)
        f0s.append(f0)

    prevention_df['geometry'] = burn_candidates
    prevention_df['area'] = f0s
    prevention_df['burned_area'] = f1s
    prevention_df['bldg'] = f2s
    prevention_df['habitat'] = f3s
    prevention_df['hazard'] = f4s
    prevention_df['intensity'] = f5s

    prevention_df['covered_raster_ids'] = covered_ids

    return prevention_df


if __name__ == '__main__':
# check if there is a config file in the arguments
    if len(sys.argv) > 1:
        config_file_name = sys.argv[1]
    else:
        config_file_name = 'config.yaml'

    # load the paths to files from yaml file
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
        rx_burn_units_path = config['rx_burn_units_path']
        values_file_path = config['values_file_path']
        prevention_file_path = config['prevention_file_path']
        results_dir = config['results_dir']
        full_ignitions_file_path = config['full_ignitions_file_path']
        burned_area_dir = config['burned_area_dir']
        bldg_dmg_dir = config['bldg_dmg_dir']
        habitat_dmg_dir = config['habitat_dmg_dir']
        intensity_dir = config['intensity_dir']
        budget = config['budget']
        

    intensity_file_names = glob(os.path.join(intensity_dir, '*.tif'))
    hazard_file_names = glob(os.path.join(results_dir, 'hazards/*.tif'))
    total_files = len(intensity_file_names)
    print('total simulated burns =', total_files)
    
    complete_lb, complete_rb, complete_tb, complete_bb = [753003.071258364, 839057.6399108863, 4133476.546731642, 4230634.714689108]
    transform = Affine(10, 0.0, complete_lb, 
                    0.0, -10, complete_bb)
    
    full_ignitions_df = pd.read_csv(full_ignitions_file_path)

    hazard_raster_path = os.path.join(results_dir, 'merged.tif')

    burn_prob_raster_path = os.path.join(results_dir, 'raster_burn_prob.tif')
    burn_prob_raster = rasterio.open(burn_prob_raster_path)
    burn_prob_values = burn_prob_raster.read(1)
    xmin, ymin, xmax, ymax = burn_prob_raster.bounds

    # hazard_raster_path = os.path.join(results_dir, 'merged.tif')
    hazard_raster = rasterio.open(hazard_raster_path)
    window = from_bounds(xmin, ymin, xmax, ymax, transform=hazard_raster.transform)
    hazard_values = hazard_raster.read(1, window=window)
    pad_width = ((0, burn_prob_values.shape[0] - hazard_values.shape[0]), (0, burn_prob_values.shape[1] - hazard_values.shape[1]))
    hazard_values = np.pad(hazard_values, pad_width, mode='constant', constant_values=0.0)
    print(window, hazard_values.shape)

    initial_hazard = hazard_values * burn_prob_values
    
    with rasterio.open(os.path.join(results_dir, 'initial_Hazard.tif'),
                            'w',
                            driver='GTiff',
                            height = initial_hazard.shape[0],
                            width = initial_hazard.shape[1],
                            count=1,
                            dtype=initial_hazard.dtype,
                            crs=burn_prob_raster.crs,
                            transform = transform) as dst:
                dst.write(initial_hazard, 1)
                dst.close()

    values_df = get_burn_area_values(values_file_path, hazard_raster_path)
    write_csv_to_file(values_file_path, values_df)

    prevention_df = generate_prevention_df(rx_burn_units_path, values_df)
    write_csv_to_file(prevention_file_path, prevention_df)
    