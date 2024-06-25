#!/usr/bin/env python3

import argparse
import math
import os
import sys
import time
import yaml

from dask.dataframe import concat
from dask.distributed import Client, LocalCluster
from rasterio.transform import Affine
from shapely.geometry import box
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rioxarray as rxr

CPU = 2

def process_files(file_names, lb, rb, tb, bb, client):
    
    #start_time = time.time()

    cnt = 0
    
    df_concat_list = []
    
    for file_name in file_names:
        with rxr.open_rasterio(file_name) as raster:
            
            raster = raster.rename('intensity')
            cur_df = raster.to_dask_dataframe()
    
        if cur_df is not None:
            df_concat_list.append(cur_df)
        cnt += 1
        if cnt % 200 == 0:
            print('.', end='', flush=True)
            #break
   
    #print('load time', int(time.time() - start_time), 's', end=' ', flush=True)
    #start_time = time.time()

    df = concat(df_concat_list)
    
    # remove columns we don't need
    df = df.drop(columns=['spatial_ref', 'band'])

    # keep non-zero intensities
    df = df.where(df['intensity'] > 0) 

    # remove outside 
    df = df.where((df['x'] > lb) & (df['x'] < rb) & (df['y'] > tb) & (df['y'] < bb))

    df = df.dropna()
    
    df['x'] = ((df['x'].astype(float) - lb) // 10).astype(int)
    df['y'] = ((df['y'].astype(float) - tb) // 10).astype(int)

    #print('preproc time', int(time.time() - start_time), 's', end=' ', flush=True)
    start_time = time.time()

    group_series = df.groupby(['x', 'y']).intensity.apply(pd.Series.to_numpy, meta=('i', 'f8')).repartition(npartitions=CPU)
    percentile_df = group_series.map(lambda r: np.quantile(r, 0.95), meta=('q', 'f8')).compute()
    
    print('calc time', int(time.time() - start_time), 's')

    del df
    del group_series

    return percentile_df

def start(config_file_name, dask_scheduler_url, max=None):
    
    # load the paths to files from yaml file
    with open(config_file_name, 'r') as f:
        config = yaml.safe_load(f)
        results_dir = config['results_dir']
        intensity_dir = config['intensity_dir']
        budget = config['budget']
        lb = config['lb'] - 10
        rb = config['rb'] + 10
        tb = config['tb'] - 10
        bb = config['bb'] + 10

    lock_file_name = os.path.join(results_dir, 'hazards/', 'subchunks_{}_{}_{}_{}.lock'.format(lb, rb, tb, bb))
    out_file_name = os.path.join(results_dir, 'hazards/', 'subchunks_{}_{}_{}_{}.tif'.format(lb, rb, tb, bb))

    if os.path.exists(lock_file_name):
        print(config_file_name, 'lock exists; skip.')
        return

    if os.path.exists(out_file_name):
        print(config_file_name, 'output exists; skip.')
        return
   
    with open(lock_file_name, 'w') as f:
        pass

    try:
        bounds_df = gpd.read_file(f"{intensity_dir}/bounds.shp",
                                  mask = box(lb, tb, rb, bb))
    
        intensity_file_names = [f"{intensity_dir}/{fname}" for fname in bounds_df['location']]

        if len(intensity_file_names) == 0:
            print('WARNING: no rasters; skip.', config_file_name)
            return
   
        if max is not None and len(intensity_file_names) > max:
            print('WARNING: raster > max; skip.', config_file_name)
            return

        with rasterio.open(intensity_file_names[0]) as raster:
            crs = raster.crs
            
        total_files = len(intensity_file_names)
        print(config_file_name, 'burns =', total_files, end=' ', flush=True)
    
        with Client(dask_scheduler_url) as client:
            p_df = process_files(intensity_file_names, lb, rb, tb, bb, client)
        
    
        transform = Affine(10, 0.0, lb, 0.0, -10, bb)
    
        heat_array_area = np.zeros([math.ceil(math.ceil(bb - tb) / 10),
                                   math.ceil(math.ceil(rb - lb) / 10)])
    
        heat_array_intensity = np.zeros_like(heat_array_area)


        for item in p_df.items():
            x, y = item[0]
            val = item[1]
            try:
                heat_array_intensity[y][x] = val
            except Exception as e:
                print('error at', x, y)
                raise e

        del p_df

        heat_array_intensity = np.flipud(heat_array_intensity)

        with rasterio.open(out_file_name,
                           'w',
                           driver='GTiff',
                           height = heat_array_area.shape[0],
                           width = heat_array_area.shape[1],
                           count=1,
                           dtype=heat_array_area.dtype,
                           crs=crs,
                           transform = transform) as dst:
    
                dst.write(heat_array_intensity, 1)

    finally:
        os.remove(lock_file_name)


if __name__ == '__main__':
   
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', required=True, help='dask scheduler url')
    parser.add_argument('-c', required=True, nargs='+', help='subchunk config file(s)')
    parser.add_argument('-m', required=False, type=int, help='max rasters')

    args = parser.parse_args()
   
    for c in args.c:
        start(c, args.s, max=args.m)
