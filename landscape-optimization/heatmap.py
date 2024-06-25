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
shapely.speedups.enable()
import time
import yaml
import sys

import math
from rasterio.merge import merge
import shutil
from rasterio.windows import Window
from rasterio.windows import from_bounds
from rasterio.transform import Affine
from rasterio.plot import show
from shapely.geometry import box
from time import time_ns
from scipy.sparse import csr_matrix

import utils
import subprocess
import sys

if __name__ == '__main__':
    # check if there is a config file in the arguments
    if len(sys.argv) > 1:
        config_file_name = sys.argv[1]
    else:
        config_file_name = 'config.yaml'

    # load the paths to files from yaml file
    with open(config_file_name, 'r') as f:
        config = yaml.safe_load(f)
        rx_burn_units_path = config['rx_burn_units_path']
        values_file_path = config['values_file_path']
        prevention_file_path = config['prevention_file_path']
        results_dir = config['results_dir']
        full_ignitions_file_path = config['full_ignitions_file_path']
        burned_area_dir = config['burned_area_dir']
        bldg_dmg_dir = config['bldg_dmg_dir']
        habitat_dmg_dir = config['habitat_dmg_dir']
        budget = config['budget']
        complete_lb = config['left_bound']
        complete_rb = config['right_bound']
        complete_tb = config['top_bound']
        complete_bb = config['bottom_bound']
        alpha = config['alpha']

    # Check if the budget is a single number or a list containing multiple budgets, 
    if isinstance(budget, list):
        # generate new configuration files for each budget, and create subprocesses for each budget to run them in parallel
        # check if the new folder for configuration file for different budget exist, if not create it
        if not os.path.exists(os.path.join('config_files_diff_budgets')):
            os.makedirs(os.path.join('config_files_diff_budgets'))
        procs = []
        for b in budget:
            config['budget'] = b
            # change the results_dir to a new folder for each budget
            config['results_dir'] = os.path.join(results_dir, 'budget_' + str(b))
            # check if the new folder for results for different budget exist, if not create it
            if not os.path.exists(config['results_dir']):
                print('Error: Please run the main optimization first to generate the results for different budgets.')
            with open(os.path.join('config_files_diff_budgets', 'config_' + str(b) + '.yaml'), 'w') as f:
                yaml.dump(config, f)
            proc = subprocess.Popen(['python', 'heatmap.py', os.path.join('config_files_diff_budgets', 'config_' + str(b) + '.yaml')])
            procs.append(proc)
        for proc in procs:
            proc.wait()
        sys.exit()

    burned_area_tifs = glob(os.path.join(burned_area_dir, '*.tif'))
    total_files = len(burned_area_tifs)
    print('total simulated burns =', len(burned_area_tifs))

    values_df = pd.read_csv(values_file_path)

    def converter(instr):
        return np.fromstring(instr[1:-1],sep=',')
    up_prevention_df = pd.read_csv(prevention_file_path, converters = {'covered_raster_ids': converter})
    prevention_df = up_prevention_df

    solutions_path = os.path.join(results_dir, 'solutions.csv')
    # read result_subsets from solutions.csv
    solutions_df = pd.read_csv(solutions_path, index_col=False, header=None)
    
    result_subsets = []
    for i in range(len(solutions_df)):
        values = np.array(solutions_df.iloc[i])
        # parse nan and -1 from solutions.csv
        values = values[~np.isnan(values)]
        values = values[values != -1]
        result_subsets.append(values)

    heatmap_res_dir = results_dir
    if not os.path.exists(heatmap_res_dir):
        os.makedirs(heatmap_res_dir)
    
    if not os.path.exists(os.path.join(heatmap_res_dir, 'heatmaps')):
        os.makedirs(os.path.join(heatmap_res_dir, 'heatmaps'))

    transform = Affine(10, 0.0, complete_lb, 
                    0.0, -10, complete_bb)

    skipped = 0

    plan_heat_array = []
    plan_heat_array_bldg = []
    plan_heat_array_habi = []

    used_mask = np.zeros([len(prevention_df)], dtype=bool)


    for i,plan in enumerate(result_subsets):
        for plan_idx in plan.astype(int):
            used_mask[plan_idx]=True

    xdim = int(math.ceil(complete_bb - complete_tb))//10
    ydim = int(math.ceil(complete_rb - complete_lb))//10

    for poly_idx in range(len(prevention_df)):
        if used_mask[poly_idx] == False:
            plan_heat_array.append([])
            plan_heat_array_bldg.append([])
            plan_heat_array_habi.append([])
            continue

        heat_array = np.zeros([xdim, ydim])
        heat_array_bldg = np.zeros([xdim, ydim])
        heat_array_habi = np.zeros([xdim, ydim])
        print('heat array init done')
        poly = prevention_df.iloc[poly_idx].geometry

        for raster_num in prevention_df.iloc[poly_idx]['covered_raster_ids'].astype(int):
            raster_row = values_df.iloc[raster_num]
            file_name = raster_row['filename']
            # print(file_name)

            raster = rasterio.open(os.path.join(burned_area_dir, 'burned_area-' + file_name + '.tif'))
            xmin, ymin, xmax, ymax = raster.bounds
            img = raster.read(1)
            try:
                heat_array[int(-ymax+complete_bb)//10:int((-ymax+complete_bb)//10+(len(img))), int(xmin-complete_lb)//10:int((xmin-complete_lb)//10+(len(img[0])))] += img
            except:
                skipped += 1
                continue
            raster.close()

            if raster_row['bldg_dmg'].astype(int) > 0:
                raster = rasterio.open(os.path.join(bldg_dmg_dir, 'building_damage-' + file_name + '.tif'))
                xmin, ymin, xmax, ymax = raster.bounds
                img = raster.read(1)
                try:
                    heat_array_bldg[int(-ymax+complete_bb)//10:int((-ymax+complete_bb)//10+(len(img))), int(xmin-complete_lb)//10:int((xmin-complete_lb)//10+(len(img[0])))] += img
                except:
                    skipped += 1
                    continue
                raster.close()

            if raster_row['habitat_dmg'].astype(int) > 0:
                raster = rasterio.open(os.path.join(habitat_dmg_dir, 'habitat_damage-' + file_name + '.tif'))
                xmin, ymin, xmax, ymax = raster.bounds
                img = raster.read(1)
                try:
                    heat_array_habi[int(-ymax+complete_bb)//10:int((-ymax+complete_bb)//10+(len(img))), int(xmin-complete_lb)//10:int((xmin-complete_lb)//10+(len(img[0])))] += img
                except:
                    skipped += 1
                    continue
                raster.close()
        
        plan_heat_array.append(csr_matrix(heat_array))
        plan_heat_array_bldg.append(csr_matrix(heat_array_bldg))
        plan_heat_array_habi.append(csr_matrix(heat_array_habi))
    
    if results_dir[-1] != '/':
        init_hazard_file_path = os.path.join(os.path.dirname(results_dir), 'initial_Hazard.tif')
    else:
        init_hazard_file_path = os.path.join(os.path.dirname(results_dir[:-1]), 'initial_Hazard.tif')

    raster = rasterio.open(init_hazard_file_path) 
    raster_values = raster.read(1)
    xmin, ymin, xmax, ymax = raster.bounds

    if results_dir[-1] != '/':
        hazard_raster_path = os.path.join(os.path.dirname(results_dir), 'merged.tif')
    else:
        hazard_raster_path = os.path.join(os.path.dirname(results_dir[:-1]), 'merged.tif')
    hazard_raster = rasterio.open(hazard_raster_path)
    # print(hazard_raster.bounds)
    window = from_bounds(xmin, ymin, xmax, ymax, transform=hazard_raster.transform)
    hazard_values = hazard_raster.read(1, window=window)
    pad_width = ((0, raster_values.shape[0] - hazard_values.shape[0]), (0, raster_values.shape[1] - hazard_values.shape[1]))
    hazard_values = np.pad(hazard_values, pad_width, mode='constant', constant_values=0.0)

    for i,plan in enumerate(result_subsets[0:]):
        # print(i)
        start = time_ns()
        heat_array_burn_prob = alpha / total_files * np.sum([plan_heat_array[idx] for idx in plan.astype(int)], axis=0).toarray()
        heat_array = heat_array_burn_prob * hazard_values
        heat_array_bldg = alpha / total_files * np.sum([plan_heat_array_bldg[idx] for idx in plan.astype(int)], axis=0).toarray()
        heat_array_habi = alpha / total_files * np.sum([plan_heat_array_habi[idx] for idx in plan.astype(int)], axis=0).toarray()
        
        print("Plan ", i, " - Skipped: ", skipped)
        
        with rasterio.open(os.path.join(heatmap_res_dir, 'heatmaps/Hazard_{}.tif'.format(i+1)),
                                'w',
                                driver='GTiff',
                                height = heat_array.shape[0],
                                width = heat_array.shape[1],
                                count=1,
                                dtype=heat_array.dtype,
                                crs=raster.crs,
                                transform = transform,
                                compress = 'ZSTD') as dst:
                    dst.write(heat_array / total_files, 1)
                    dst.close()
        # raster.close()
        with rasterio.open(os.path.join(heatmap_res_dir, 'heatmaps/Habitat_Dmg_{}.tif'.format(i+1)),
                                'w',
                                driver='GTiff',
                                height = heat_array.shape[0],
                                width = heat_array.shape[1],
                                count=1,
                                dtype=heat_array.dtype,
                                crs=raster.crs,
                                transform = transform,
                                compress = 'ZSTD') as dst:
                    dst.write(heat_array_habi, 1)
                    dst.close()
        with rasterio.open(os.path.join(heatmap_res_dir, 'heatmaps/Bldg_Dmg{}.tif'.format(i+1)),
                                'w',
                                driver='GTiff',
                                height = heat_array.shape[0],
                                width = heat_array.shape[1],
                                count=1,
                                dtype=heat_array.dtype,
                                crs=raster.crs,
                                transform = transform,
                                compress = 'ZSTD') as dst:
                    dst.write(heat_array_bldg, 1)
                    dst.close()
        
        
        end = time_ns() - start
        print("Time taken: " + str(end/(10**9)/60) + " mins")
