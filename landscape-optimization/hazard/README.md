
1. run python ./hazard/create_subchunks.py

2. create bounds index of flamelen tifs (in sub_data/intensity):
    ```
    ls | grep tif | xargs -n1000 gdaltindex bounds.shp
    ```

3. start dask
    ```
    dask scheduler --port 8786
    dask worker --nthreads 1 --nworkers 64 --memory-limit 1.8GB tcp://localhost:8786
    ```
4. run preprocess_hazard_subtrunks.py for each config yml in results/prototype_run/hazard_config/
    - set CPU to appropriate value. I found best is number of physical cores.
    - each run will generate tif in results/hazards/
    - need to set large number of open files with `ulimit` otherwise it gets too
      many open files error.
    ```
    ulimit -n 100000
    find results/prototype_run/hazard_config/ -type f | sort | xargs python ./hazard/preprocess_hazard_subtrunks.py -s tcp://localhost:8786 -c
    ```

5. merge results with gdal_merge.py
    ```
    python hazard/gdal_merge.py -o results/prototype_run/merged.tif results/prototype_run/hazards/subchunks*.tif
    ```
    where {results_dir} should be replaced by the results_dir specified in the ```config.yaml```.
