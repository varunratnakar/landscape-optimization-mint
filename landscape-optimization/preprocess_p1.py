import math
import os
import shutil
from collections import defaultdict
from glob import glob
# from pathlib import Path
from time import time_ns

import yaml
import geopandas as gpd
#from pymoo.indicators import get_performance_indicator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
import shapely.speedups
from pymoo.algorithms.moo.ctaea import CTAEA
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.core.callback import Callback
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.problem import ElementwiseProblem, Problem
from pymoo.core.sampling import Sampling
#from pymoo.factory import get_visualization
from pymoo.indicators.hv import HV
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.problems.many import get_ref_dirs
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter
from rasterio.merge import merge
from rasterio.plot import show
from rasterio.transform import Affine
from rasterio.windows import Window
from shapely.geometry import MultiPolygon, Point, box
from shapely.geometry.polygon import Polygon

import utils

shapely.speedups.enable()

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

def get_burn_area_values(truncated_ignitions_df, burned_area_dir, bldg_dmg_file_names, habitat_dmg_file_names, config):
    values_df = pd.DataFrame(columns = ['filename', 'x_ignition', 'y_ignition', 
                                    'burn_area', 'bldg_dmg', 'habitat_dmg', 'xmin', 'ymin', 'xmax', 'ymax'])
    values_df[['filename', 'x_ignition', 'y_ignition']] =  truncated_ignitions_df[['filename', 'x_ignition', 'y_ignition']]

    burn_area_values = []
    bld_dmg_values = []
    habitat_dmg_values = []

    all_xmin, all_ymin, all_xmax, all_ymax = np.inf, np.inf, 0, 0

    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []
    

    for name in values_df.filename:
        file_name = os.path.join(burned_area_dir,'burned_area-{}.tif'.format(name))
        raster = rasterio.open(file_name)
        xmin, ymin, xmax, ymax = raster.bounds
        # maintain the bounds
        all_xmin = min(all_xmin, xmin)
        all_ymin = min(all_ymin, ymin)
        all_xmax = max(all_xmax, xmax)
        all_ymax = max(all_ymax, ymax)
        xmins.append(xmin)
        ymins.append(ymin)
        xmaxs.append(xmax)
        ymaxs.append(ymax)

        img = raster.read(1)
        burn_val = np.sum(img)
        burn_area_values.append(burn_val)
    burn_area_values = np.array(burn_area_values)

    # check does config has key left_bound
    if 'left_bound' not in config:
        config['left_bound'] = all_xmin
        config['right_bound'] = all_xmax
        config['bottom_bound'] = all_ymax
        config['top_bound'] = all_ymin
        # write back to config so main and heatmap can use it
        with open('config.yaml', 'w') as f:
            yaml.dump(config, f)


    values_df['burn_area'] = burn_area_values.tolist()
    values_df['xmin'] = xmins
    values_df['ymin'] = ymins
    values_df['xmax'] = xmaxs
    values_df['ymax'] = ymaxs

    bld_dmg_values = np.zeros(shape = burn_area_values.shape)

    error_bldg_firenums = []

    for file_name in bldg_dmg_file_names:
    # file_name = os.path.join(burned_area_dir,'burned_area-{}.tif'.format(name))
        # fire_number = remove_prefix(os.path(file_name).stem, 'building_damage-intensity-')
        fire_number = remove_prefix(os.path.splitext(os.path.basename(file_name))[0], 'building_damage-')
        if fire_number not in values_df.filename.values:
            error_bldg_firenums.append(fire_number)
            continue
        fire_index = np.where(values_df.filename == fire_number)[0][0]

        raster = rasterio.open(file_name)
        img = raster.read(1)
        bld_dmg_val = np.sum(img)
        bld_dmg_values[fire_index] = bld_dmg_val
    
    if len(error_bldg_firenums) > 0:
        print('Error building intensity fire numbers')
        print(error_bldg_firenums)
    
    values_df['bldg_dmg'] = bld_dmg_values.tolist() 

    habitat_dmg_values = np.zeros(shape = burn_area_values.shape)
    ct = 1
    for file_name in habitat_dmg_file_names:
    # file_name = os.path.join(burned_area_dir,'burned_area-{}.tif'.format(name))
        # fire_number = remove_prefix(os.path(file_name).stem, 'habitat_damage-')
        fire_number = remove_prefix(os.path.splitext(os.path.basename(file_name))[0], 'habitat_damage-')
        # fire_index = np.where(values_df.filename == fire_number)[0][0]
        fire_index = values_df.index[values_df['filename'] == fire_number].tolist()[0]
        try:
            raster = rasterio.open(file_name)
            img = raster.read(1)
            habitat_dmg_val = np.sum(img)
        except:
            print(fire_number)
            habitat_dmg_val = 0.0
        habitat_dmg_values[fire_index] = habitat_dmg_val

    values_df['habitat_dmg'] = habitat_dmg_values.tolist()
    
    return values_df

def write_csv_to_file(file_path, data):
    data.to_csv(file_path, index=False)

def point_in_poly(point, polygon):
    return polygon.contains(Point(point))

def generate_prevention_df(rx_burn_units_path, values_df):
    prevention_df = pd.DataFrame(columns=['geometry','f1', 'f2', 'f3', 'covered_raster_ids'])

    func = lambda x, y : point_in_poly(x, y)
    vector_func = np.vectorize(func)

    f1s = []
    f2s = []
    f3s = []
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
        
        covered_ids.append(list(covered))
        f1s.append(f1)
        f2s.append(f2)
        f3s.append(f3)

    prevention_df['geometry'] = burn_candidates
    prevention_df['f1'] = f1s
    prevention_df['f2'] = f2s
    prevention_df['f3'] = f3s
    prevention_df['covered_raster_ids'] = covered_ids

    return prevention_df

if __name__ == "__main__":
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
        budget = config['budget']

    # Preprocessing 
    full_ignitions_df = pd.read_csv(full_ignitions_file_path)
    # full_ignitions_df = []
    burn_file_names = glob(os.path.join(burned_area_dir, '*.tif'))
    bldg_dmg_file_names = glob(os.path.join(bldg_dmg_dir, '*.tif'))
    habitat_dmg_file_names = glob(os.path.join(habitat_dmg_dir, '*.tif'))

    burned_area_tifs = glob(os.path.join(burned_area_dir, '*.tif'))
    total_files = len(burned_area_tifs)
    print('total simulated burns =', len(burned_area_tifs))

    bldg_damage_tifs = glob(os.path.join(bldg_dmg_dir, '*.tif'))
    habi_damage_tifs = glob(os.path.join(habitat_dmg_dir, '*.tif'))

    print("Run main function")

    ## Call when values_df needs to be created
    truncated_ignitions_df = get_truncated_ignitions(full_ignitions_df, burn_file_names)
    values_df = get_burn_area_values(truncated_ignitions_df, burned_area_dir, bldg_dmg_file_names, habitat_dmg_file_names, config)
    write_csv_to_file(values_file_path, values_df)

    # ## Call to generate prevention_df
    prevention_df = generate_prevention_df(rx_burn_units_path, values_df)
    write_csv_to_file(prevention_file_path, prevention_df)

    print("Generated prevention df")

    complete_lb = config['left_bound']
    complete_bb = config['bottom_bound']
    complete_rb = config['right_bound']
    complete_tb = config['top_bound']
    heat_array_area = np.zeros((int(complete_bb - complete_tb) // 10, int(complete_rb - complete_lb) // 10))
    heat_array_bldg = np.zeros_like(heat_array_area)
    heat_array_habi = np.zeros_like(heat_array_area)

    transform = Affine(10, 0.0, complete_lb, 
                    0.0, -10, complete_bb)
    
    skipped = 0

    for file_name in burn_file_names:
        try:
            raster = rasterio.open(file_name)
            xmin, ymin, xmax, ymax = raster.bounds
            img = raster.read(1)
        
            heat_array_area[int(-ymax+complete_bb)//10:int((-ymax+complete_bb)//10+(len(img))), int(xmin-complete_lb)//10:int((xmin-complete_lb)//10+(len(img[0])))] += img
        except:
            skipped += 1
            continue
        raster.close() 
    
    raster = rasterio.open(burn_file_names[1])
    # # print(raster.res)
    
    with rasterio.open(os.path.join(results_dir, 'summed_raster_heatmap_area.tif'),
                            'w',
                            driver='GTiff',
                            height = heat_array_area.shape[0],
                            width = heat_array_area.shape[1],
                            count=1,
                            dtype=heat_array_area.dtype,
                            crs=raster.crs,
                            transform = transform) as dst:
                dst.write(heat_array_area, 1)
                dst.close()
    with rasterio.open(os.path.join(results_dir, 'raster_burn_prob.tif'),
                            'w',
                            driver='GTiff',
                            height = heat_array_area.shape[0],
                            width = heat_array_area.shape[1],
                            count=1,
                            dtype=heat_array_area.dtype,
                            crs=raster.crs,
                            transform = transform) as dst:
                dst.write(heat_array_area / total_files, 1)
                dst.close()
    
    # print('print burn prob rasters.')

    for file_name in bldg_dmg_file_names:
        try:
            raster = rasterio.open(file_name)
            xmin, ymin, xmax, ymax = raster.bounds
            
            img = raster.read(1)
        
            heat_array_bldg[int(-ymax+complete_bb)//10:int((-ymax+complete_bb)//10+(len(img))), int(xmin-complete_lb)//10:int((xmin-complete_lb)//10+(len(img[0])))] += img
        except:
            skipped += 1
            continue
        raster.close()

    for file_name in habitat_dmg_file_names:
        try:
            raster = rasterio.open(file_name)
            xmin, ymin, xmax, ymax = raster.bounds
            # print(xmin, ymin, xmax, ymax)
            img = raster.read(1)
            # img = np.flip(img, 0)
            # print(img.shape, raster.bounds)
        # try:
            heat_array_habi[int(-ymax+complete_bb)//10:int((-ymax+complete_bb)//10+(len(img))), int(xmin-complete_lb)//10:int((xmin-complete_lb)//10+(len(img[0])))] += img
        except:
            skipped += 1
            continue
        raster.close()

    # raster.close()
    with rasterio.open(os.path.join(results_dir, 'initial_Habitat_Dmg.tif'),
                            'w',
                            driver='GTiff',
                            height = heat_array_area.shape[0],
                            width = heat_array_area.shape[1],
                            count=1,
                            dtype=heat_array_area.dtype,
                            crs=raster.crs,
                            transform = transform) as dst:
                dst.write(heat_array_habi / total_files, 1)
                dst.close()

    with rasterio.open(os.path.join(results_dir, 'initial_Bldg_Dmg.tif'),
                            'w',
                            driver='GTiff',
                            height = heat_array_area.shape[0],
                            width = heat_array_area.shape[1],
                            count=1,
                            dtype=heat_array_area.dtype,
                            crs=raster.crs,
                            transform = transform) as dst:
                dst.write(heat_array_bldg / (total_files * 100), 1)
                dst.close()
