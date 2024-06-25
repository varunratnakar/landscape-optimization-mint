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
import shapely.speedups
from shapely.geometry import MultiPolygon
import time
import yaml
import dask
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
import csv
import rioxarray as rxr
import xarray as xr

from datetime import datetime

# def raster_to_df_dask(file_name):
#     # read the raster file and convert to a dask dataframe
#     raster = rasterio.open(file_name)
#     xmin, ymin, xmax, ymax = raster.bounds
#     img = raster.read(1)
#     cnt += 1
#     loc_names = []
#     values = []
#     for i in range(len(img)):
#         for j in range(len(img[0])):
#             if img[i][j] > 0:
#                 loc_names.append(str([int(-ymax+complete_bb)//10 + i, int(xmin-complete_lb)//10 + j]))
#                 values.append(img[i][j])
#     raster.close()
#     tmp_df = pd.DataFrame({'loc': loc_names, 'intensity': values})
#     cur_df = from_pandas(tmp_df, chunksize=10000)
#     return cur_df

def raster_to_df_rioxarray(file_name):
    # read the raster file and convert to an xarray dataframe
    raster  = rxr.open_rasterio(file_name, chunk = 100) # 100 is the chunk size in the x and y direction
    raster = raster.rename('intensity')
    # print(raster)
    df = raster.to_dask_dataframe()
    # print(df.head())
    raster.close()
    df = df.where(df['intensity'] > 0)
    # print(df.head())
    df = df.dropna()
    # print(df.head())
    # df['loc'] = df['x'].astype(str) + ',' + df['y'].astype(str)
    df['loc'] = (df['x'].astype(float) // 10).astype(str) + ',' + (df['y'].astype(float) // 10).astype(str)
    # print(df.head())
    df = df.drop(columns = ['x', 'y'])
    # print(df.head())
    df = df.reset_index()
    # print(df.head())
    return df

complete_lb, complete_rb, complete_tb, complete_bb = [753003.071258364, 839057.6399108863, 4133476.546731642, 4230634.714689108]
def update_all_rasters_heap(df, file_names):
    # update the heap with the intensity values of the raster files
    exception_file_list = []
    cnt = 0
    print('In total file num:', len(file_names))

    # df = dd.from_dict({}, npartitions=1)
    df_concat_list = []
    # file = open('dfs.csv', 'a+', newline ='')
    for file_name in file_names:
        # read the raster file and convert to an dask dataframe
        # 100 is the chunk size in the x and y direction
        with rxr.open_rasterio(file_name, chunk = 10) as raster: 
            cur_df = raster.rename('intensity').to_dask_dataframe()
            raster.close()
            
        df_concat_list.append(cur_df)

        # cur_df = raster_to_df_rioxarray(file_name)
        # dd.to_csv(cur_df, 'data/dfs1.csv', single_file=True, mode='a', compute=False)
        # dd.to_csv(cur_df, 'data/dfs2.csv', single_file=True, mode='a', compute=True)
        # percentile_df = cur_df.groupby('loc')['intensity'].apply(lambda group: np.percentile(group, 95), meta=('intensity', 'f8'))
        # df_concat_list.append(percentile_df)
        # df_concat_list.append(cur_df)
        # df = dd.concat([cur_df])
        cnt += 1
        if cnt % 1000 == 0:
            print('finish', cnt, 'file')
        # if cnt == 10:
        #     break
    print("ABC")
    df = concat(df_concat_list)
    df = df.where(df['intensity'] > 0)
    df = df.dropna()
    df['loc'] = ((df['x'].astype(float) - complete_lb) // 10).astype(str) + ',' + ((complete_bb - df['y'].astype(float)) // 10).astype(str)
    df = df.drop(columns = ['x', 'y', 'band', 'spatial_ref'])
    # df = df.reset_index()
    df = df.set_index('loc')
    # print(df.head())
    print(df.npartitions, 'p1')
    percentile_df = df.map_partitions(lambda group: np.percentile(group, 95), meta=('intensity', 'f8'))
    # percentile_df = df.groupby('loc')['intensity'].apply(lambda group: np.percentile(group, 95), meta=('intensity', 'f8'))
    print(percentile_df.npartitions, 'p2')
    percentile_df = percentile_df.compute()
    # print(percentile_df)
    return percentile_df
    # print(df_concat_list)
    # return None
    # df = concat(df_concat_list)
    # print("123")
    # print(df.compute())
    # return None
    # calculate the percentile of the intensity values and return
    # percentile_df = df.groupby('loc')['intensity'].sum()
    # percentile_df = df.groupby('loc')['intensity'].apply(lambda group: np.percentile(group, 95), meta=('intensity', 'f8'))
    # percentile_df = df.groupby('loc').apply(lambda group: np.percentile(group['intensity'], 75), meta=('intensity', 'f8'))
    # print("XYZ")
    # df.to_csv('data/dask')
    # print('after csv')
    # return df.compute()
    # print(percentile_df.memory_usage_per_partition)
    # print(percentile_df.npartitions, 'partitions')
    # percentile_df.repartition(npartitions=1)
    # print(percentile_df.memory_usage_per_partition)
    # print(percentile_df.npartitions, 'partitions')
    # percentile_df.to_csv('data/df.csv', single_file=True)
    # print("after compute")
    # percentile_df = df.compute()
    # percentile_df = percentile_df.compute()
    # print(percentile_df)
    # return percentile_df
    # print("here!")
    # return dd.concat(df_concat_list).groupby('loc')['intensity'].apply(lambda group: np.percentile(group, 95), meta=('intensity', 'f8')).compute()

if __name__ == '__main__':
    startTime = datetime.now()
    # exit(1)
# check if there is a config file in the arguments
    # if len(sys.argv) > 1:
    #     config_file_name = sys.argv[1]
    # else:
    #     config_file_name = 'config.yaml'

    # load the paths to files from yaml file
    config_file_name = 'config.yaml'
    with open(config_file_name, 'r') as f:
        config = yaml.safe_load(f)
        # rx_burn_units_path = config['rx_burn_units_path']
        results_dir = config['results_dir']
        intensity_dir = config['intensity_dir']
        # budget = config['budget']
    # print(rx_burn_units_path, results_dir, intensity_dir, budget)
    # exit(1)
    intensity_file_names = glob(os.path.join(intensity_dir, '*.tif'))
    total_files = len(intensity_file_names)
    print('total simulated burns =', total_files)
    
    # worker_kwargs = {
    # 'memory_limit': '5G',
    # "distributed.worker.memory.target": 0.9,
    # "distributed.worker.memory.spill": 0.925,
    # "distributed.worker.memory.pause": 0.95
    # 'memory_target_fraction': 0.9,
    # 'memory_spill_fraction': 0.925,
    # 'memory_pause_fraction': 0.95,
    # 'memory_terminate_fraction': 0.95,
    # }
    dask.config.set({
        'distributed.memory.worker.target': 0.8,
        'distributed.scheduler.active-memory-manager.measure':'managed',
        'distributed.worker.memory.rebalance.measure': 'managed',
        # 'distributed.worker.memory.spill': False,
        # 'distributed.worker.memory.pause': False,
        # 'distributed.worker.memory.terminate': False
        # 'interface': 'lo'
    })
    cluster = LocalCluster(n_workers=4,
                       threads_per_worker=4,
                       memory_limit='2GB')
    client = Client(cluster)
    # client.rebalance()
    
    df_heat_array = None
    percentile_df = update_all_rasters_heap(df_heat_array, intensity_file_names[:1000])
    print('here222')
    print(datetime.now() - startTime)
    
    complete_lb, complete_rb, complete_tb, complete_bb = [753003.071258364, 839057.6399108863, 4133476.546731642, 4230634.714689108]
    transform = Affine(10, 0.0, complete_lb, 0.0, -10, complete_bb)
    # exit(1)
    
    heat_array_area = np.zeros([int(math.ceil(complete_bb - complete_tb))//10,int(math.ceil(complete_rb-complete_lb))//10])
    heat_array_intensity = np.zeros_like(heat_array_area)
    heat_array_intensity = heat_array_intensity.astype(int)
    print('heat array init done')
    # exit(1)
    # cluster = LocalCluster(n_workers=2,
    #                    threads_per_worker=4,
    #                    memory_target_fraction=0.9,
    #                    memory_limit='5GB')
    # client = Client(cluster)

    # df_heat_array = None
    # percentile_df = update_all_rasters_heap(df_heat_array, intensity_file_names)
    # percentile_df = client.submit(update_all_rasters_heap, df_heat_array, intensity_file_names)
    # percentile_df = percentile_df.result()
    # print('here222')
    # exit(1)
    # heat_array_intensity = np.zeros_like(heat_array_area)
    # heat_array_intensity = heat_array_intensity.astype(int)

    for i in range(len(heat_array_intensity)):
          for j in range(len(heat_array_intensity[0])):
            try:
                heat_array_intensity[i][j] = percentile_df.loc[str([i, j])]['intensity'] 
            except:
                heat_array_intensity[i][j] = 0
    print('here33!')
    print(datetime.now() - startTime)
    
    raster = rasterio.open(intensity_file_names[1])
    with rasterio.open(os.path.join(results_dir, '95_percentile_intensity.tif'),
                            'w',
                            driver='GTiff',
                            height = heat_array_area.shape[0],
                            width = heat_array_area.shape[1],
                            count=1,
                            dtype=heat_array_area.dtype,
                            crs=raster.crs,
                            transform = transform) as dst:
                dst.write(heat_array_intensity, 1)
                dst.close()
    raster.close()
    
    
    # for file_name in file_names:
    #     cur_df, crs = raster_to_df_rioxarray(file_name)
    #     percentile_df = cur_df.groupby('loc')['intensity'].apply(lambda group: np.percentile(group, 95), meta=('intensity', 'f8'))

    #     df_concat_list.append(percentile_df)
    #     cnt += 1
    #     if cnt % 100 == 0:
    #         print('finish', cnt, 'file')
    #     # if cnt == 10:
    #     #     break
    
    # print("waiting for concat")
    # df = concat(df_concat_list)
    # print("concat done.. waiting for percentile")