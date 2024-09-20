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
from preprocess_p1 import get_truncated_ignitions, get_burn_area_values

import utils
import subprocess
import sys

shapely.speedups.enable()

def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text

def write_csv_to_file(file_path, data):
    data.to_csv(file_path, index=False)

def converter(instr):
    return np.fromstring(instr[1:-1],sep=' ')

def base_optimization(values_df, budget, rx_burn_units, prevention_df):

    algorithm = NSGA2(
        pop_size = 200,
        sampling = utils.MySamplingWeight(),
        crossover = utils.BinaryCrossoverWeight(),
        mutation = utils.MyMutation(),
        eliminate_duplicates = True)
    
    problem = utils.HazardProblem(values_df, budget, rx_burn_units, prevention_df)

    res1 = minimize(problem, algorithm, ('n_gen', 1000), seed=1, verbose=True, callback = utils.GenCallback(), save_history=True)
    try:
        val200 = res1.algorithm.callback.data['gen200']
        val500 = res1.algorithm.callback.data['gen500']
        val1000 = res1.algorithm.callback.data['gen1000']
    except:
        print('Error occured when optimization. This is typically because you are setting a budget that is too small. Try something larger or report this problem to us.')
        sys.exit(1)

    res1_200 = val200[0]
    res1_500 = val500[0]
    res1_1000 = val1000[0]

    result_subsets = []
    if res1.X.any():
        for subset in res1.X:
            result_subsets.append(problem.non_zero_idx[np.where(subset)[0]])

    return res1, res1_200, res1_500, res1_1000, result_subsets

def calculate_hypervolumes(res1_200, res1_500, res1_1000):
    ref = [0 for _ in range(res1_200.shape[1])]
    hv_base = HV(ref_point = ref)
    print("hv for base_formulation at 200 gens", hv_base.do(res1_200))
    print("hv for base_formulation at 500 gens", hv_base.do(res1_500))
    print("hv for base_formulation at 1000 gens", hv_base.do(res1_1000))
    return hv_base

def plot_results(callback_val, res, save_file_path):
    
    print("Function values at gen " + callback_val + ": %s" % res)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(res[:, 0], res[:, 1], res[:, 2])
    ax.view_init(-30, 30)
    ax.set_xlabel('Burn_Area')
    ax.set_ylabel('Bldg_Dmg')
    ax.set_zlabel('Habitat_Dmg')
    plt.savefig(os.path.join(results_dir, save_file_path))
    plt.show()

def save_results(res1, res1_200, res1_500, res1_1000, result_subsets, file_paths):
    # np.savetxt(os.path.join(results_dir,file_paths['gen_res']), X = res1.F)
    obj_path = os.path.join(results_dir,file_paths['gen_res'])
    with open(obj_path, 'w') as f:
        f.write('Bldg_Dmg,Habitat_Dmg,Hazard,Cost\n')
        # print(res1.G, res1.F)
        idx = 0
        for idx in range(len(res1.F)):
            item = res1.F[idx]
            cost = res1.G[idx][0]
            # print(item, cost)
            for i in item:
                f.write("%s," % (-i))
            # print(cost, budget, cost+budget)
            f.write("%s" % (cost + budget))
            f.write("\n")

    # np.savetxt(os.path.join(results_dir,file_paths['gen_res_sub']), X = result_subsets)
    target_path = os.path.join(results_dir,file_paths['gen_res_sub'])
    # find the one that has the most item in result_subsets
    max_len = -1
    for item in result_subsets:
        if len(item) > max_len:
            max_len = len(item)

    with open(target_path, 'w') as f:
        for item in result_subsets:
            for i in item:
                f.write("%s," % i)
            for i in range(max_len - len(item)):
                f.write("-1,")
            f.write("\n")

if __name__ == "__main__":
    # check if there is a config file in the arguments
    if len(sys.argv) > 1:
        config_file_name = sys.argv[1]
    else:
        config_file_name = 'config.yaml'

    # load the paths to files from yaml file with read permissions
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
                os.makedirs(config['results_dir'])
            with open(os.path.join('config_files_diff_budgets', 'config_' + str(b) + '.yaml'), 'w') as f:
                yaml.dump(config, f)
            proc = subprocess.Popen(['python', 'main.py', os.path.join('config_files_diff_budgets', 'config_' + str(b) + '.yaml')])
            procs.append(proc)
        print(procs)
        for proc in procs:
            proc.wait()

        sys.exit()

    # Preprocessing 
    #full_ignitions_df = pd.read_csv(full_ignitions_file_path)
    full_ignitions_df = []
    burn_file_names = glob(os.path.join(burned_area_dir, '*.tif'))
    bldg_dmg_file_names = glob(os.path.join(bldg_dmg_dir, '*.tif'))
    habitat_dmg_file_names = glob(os.path.join(habitat_dmg_dir, '*.tif'))

    print("Run main function")

    values_df = pd.read_csv(values_file_path)
    print("Read Values_table from file")

    up_prevention_df = pd.read_csv(prevention_file_path, converters = {'covered_raster_ids': converter})
    prevention_df = up_prevention_df
    
    rx_burn_units = gpd.read_file(rx_burn_units_path)
    rx_burn_units = rx_burn_units.to_crs('epsg:32610')

    res1, res1_200, res1_500, res1_1000, result_subsets = base_optimization(values_df, budget, rx_burn_units, prevention_df)
    file_paths = {}

    file_paths['gen_res'] = "solutions_values.csv"
    file_paths['gen_res_sub'] = "solutions.csv"
    save_results(res1, res1_200, res1_500, res1_1000, result_subsets,file_paths)
    calculate_hypervolumes(res1_200, res1_500, res1_1000)
    
