# File Structure

- ```config.yaml```: the configuration file that specifies where the data and results will be saved to. The current example file load from ```data/```, save the preprocessed data to ```data/```, and save the final optimization results to ```results/```.
- ```preprocess.py```: pre-calculate the values and benefits for each prevention, which need to be run once before running the main.py.
- ```main.py```: the main python file that does the optimization, and output the results.
- ```heatmap.py```: the post-process code to process the results to generate benefit heatmaps.

# Data Format

While you can specify where to load the data from in the configuration file, by default, you should put the ignition points together with 3 folders of different metrics in ```data/```. In the current version of the code, we need to specify the data of burned_area, building damage, and habitat damage. Please note that each directory will be directly used as all the data for specific metrics without further parsing the name of the files, so please put different data in different folders.

# Environment and Package Requirement

Besides normal python, you need to install ***shapely***, ***numpy***, ***pymoo***, ***pandas***, ***rasterio***, ***geopandas***. You can also use the ```requirements.txt``` we provided to directly install all the packages needed. To do so, run:

```pip install -e requirements.txt```.

# Hyperparameters

Other parameters are related to data locations, please set accordingly. 

The code require the intensity files, building damage, habitat damage, and the burned area as rasters. The candidate burn units should be provided as multi-polygon shapefile, and ignition points already saved in a csv. 

# Outputs


Here is the corrected version of the sentence:

The code in this repo will output two intermediate CSV files, ```prevention_tables.csv``` and ```values.csv```, which are used by the repo itself. For later visualization, the current code will generate a set of initial damage rasters, a set of rasters describing the damage after the prescribed burn, and a pair of CSV files, ```solutions.csv``` and ```solution_values.csv```, to describe the overall information of each solution 

# Running 

For first time running, do the following step by step:
- ```python preprocess_p1.py```: this is for some initial preprocess that helps later 
- Run the code in ```hazard/```: this is for calculating the 95th percentile of all fires as an estimation of the hazard.
- ```python preprocess_p2.py```: this is preprocess for the main optimization process.
- ```python main.py```: this is the main code doing optimization.
- ```python heatmap.py```: this is to read the results from optimization, and generate heatmaps that will be used for visualization.

When running again with some change on the budget, only the last two code need to be re-run. 

