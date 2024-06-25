#!/usr/bin/env python3

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
from rasterio.transform import Affine
from rasterio.plot import show
from shapely.geometry import box
from time import time_ns
import heapq
import sys

import rioxarray as rxr
import xarray as xr

def recursive_create_subchunks(raster_value_df, lb, rb, tb, bb):
    # create subchunks of the current chunk
    # if the subchunk is still containing too much burns, then recursively create subchunks
    # if the subchunk is containing too little burns, then stop creating subchunks
    # return the list of subchunks
    parse_raster_df = raster_value_df.where(raster_value_df['x'] >= lb)
    parse_raster_df = parse_raster_df.where(parse_raster_df['x'] <= rb)
    parse_raster_df = parse_raster_df.where(parse_raster_df['y'] >= tb)
    parse_raster_df = parse_raster_df.where(parse_raster_df['y'] <= bb)
    parse_raster_df = parse_raster_df.dropna()

    if np.sum(parse_raster_df['intensity']) < 1e-5:
        return []
    elif np.sum(parse_raster_df['intensity']) * total_files < threshold :
        return [(lb, rb, tb, bb)]
    else:
        ret_list = []
        mid_x = (lb + rb) // 2
        mid_y = (tb + bb) // 2
        ret_list += recursive_create_subchunks(parse_raster_df, lb, mid_x, tb, mid_y)
        ret_list += recursive_create_subchunks(parse_raster_df, lb, mid_x, mid_y, bb)
        ret_list += recursive_create_subchunks(parse_raster_df, mid_x, rb, tb, mid_y)
        ret_list += recursive_create_subchunks(parse_raster_df, mid_x, rb, mid_y, bb)
        return ret_list

def chunk_idx_to_coordinates(chunk_list):
    ret_list_coordinates = []
    ret_list_polygons = []
    for chunk in chunk_list:
        lb, rb, tb, bb = chunk
        ret_list_polygons.append(Polygon(
            ((lb, tb), (lb, bb), (rb, bb), (rb, tb))
        ))
        
    return MultiPolygon(ret_list_polygons)

if __name__ == '__main__':
# check if there is a config file in the arguments
    if len(sys.argv) > 1:
        config_file_name = sys.argv[1]
    else:
        config_file_name = 'config.yaml'

    # load the paths to files from yaml file
    with open(config_file_name, 'r') as f:
        config = yaml.safe_load(f)
        results_dir = config['results_dir']
        intensity_dir = config['intensity_dir']
        budget = config['budget']
    threshold = 5 * (10**6)

    intensity_file_names = glob(os.path.join(intensity_dir, '*.tif'))
    total_files = len(intensity_file_names)
    print('total simulated burns =', total_files)
    
    complete_lb, complete_rb, complete_tb, complete_bb = [753003.071258364, 839057.6399108863, 4133476.546731642, 4230634.714689108]
    transform = Affine(10, 0.0, complete_lb, 
                    0.0, -10, complete_bb)
    
    burn_prob_heatmap_path = os.path.join(results_dir, 'raster_burn_prob.tif')
    # burn_prob_heatmap = rasterio.open(burn_prob_heatmap_path)
    # burn_prob_heatmap_value = burn_prob_heatmap.read(1)
    # burn_prob_heatmap.close()

    raster  = rxr.open_rasterio(burn_prob_heatmap_path) 
    raster = raster.rename('intensity')
    
    burn_prob_df = raster.to_dask_dataframe()
    raster.close()
    burn_prob_df = burn_prob_df.where(burn_prob_df['intensity'] > 0)
    burn_prob_df = burn_prob_df.dropna()
    burn_prob_df = burn_prob_df.compute() # tranforming to pandas dataframe
    # print(burn_prob_heatmap_value.shape)
    # print(np.sum(burn_prob_heatmap_value))

    # all_chunks = recursive_create_subchunks(burn_prob_heatmap_value, 0, burn_prob_heatmap_value.shape[0], 0, burn_prob_heatmap_value.shape[1])
    all_chunks = recursive_create_subchunks(burn_prob_df, complete_lb, complete_rb, complete_tb, complete_bb)
    chunk_polygons = chunk_idx_to_coordinates(all_chunks)

    #save chunk polygons to file
    chunk_polygons_gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries(chunk_polygons))
    chunk_polygons_gdf.to_file(os.path.join(results_dir, 'chunk_polygons.shp'))

    #save chunk index and coordinates to file
    chunk_coordinate_df = pd.DataFrame(all_chunks, columns=['chunk_lb', 'chunk_rb', 'chunk_tb', 'chunk_bb'])
    chunk_coordinate_df.to_csv(os.path.join(results_dir, 'chunk_coordinate.csv'), index=False)

    # if hazard subfolder in results does not exist, create it
    hazard_dir = os.path.join(results_dir, 'hazards')
    if not os.path.exists(hazard_dir):
        os.makedirs(hazard_dir)
    
    # if hazard config subfolder in results does not exist, create it
    hazard_config_dir = os.path.join(results_dir, 'hazard_config')
    if not os.path.exists(hazard_config_dir):
        os.makedirs(hazard_config_dir)
    
    # create config files for each subchunk
    for chunk in all_chunks:
        lb, rb, tb, bb = chunk
        config = {
            'lb': lb,
            'rb': rb,
            'tb': tb,
            'bb': bb,
            'budget': budget,
            'results_dir': results_dir,
            'intensity_dir': intensity_dir
        }
        with open(os.path.join(hazard_config_dir, str(lb) + '_' + str(rb) + '_' + str(tb) + '_' + str(bb) + '.yaml'), 'w') as f:
            yaml.dump(config, f)
