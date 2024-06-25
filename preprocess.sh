#!/bin/bash

FULL_IGNITIONS_FILE=""
RX_BURN_UNITS_FILE=""
BURNED_AREA_FILE=""
BUILDING_DAMAGE_FILE=""
HABITAT_DAMAGE_FILE=""
INTENSITY_FILE=""

ALPHA=0.1
BUDGET=10

while [[ $# -gt 0 ]]; do
  case $1 in
    -f | --full_ignitions_file)
        echo "Processing 'full_ignitions_file' option. Input argument is '$2'"
        FULL_IGNITIONS_FILE="$2"
        shift 2
        ;;
    -r | --rx_burn_units_file)
        echo "Processing 'rx_burn_units_file' option. Input argument is '$2'"
        RX_BURN_UNITS_FILE="$2"
        shift 2
        ;;
    -u | --burned_area_file)
        echo "Processing 'burned_area_file' option. Input argument is '$2'"
        BURNED_AREA_FILE=$2
        shift 2
        ;;
    -d | --building_damage_file)
        echo "Processing 'building_damage_file' option. Input argument is '$2'"
        BUILDING_DAMAGE_FILE=$2
        shift 2
        ;;
    -h | --habitat_damage_file)
        echo "Processing 'habitat_damage_file' option. Input argument is '$2'"
        HABITAT_DAMAGE_FILE=$2
        shift 2
        ;;
    -i | --intensity_file)
        echo "Processing 'intensity_file' option. Input argument is '$2'"
        INTENSITY_FILE=$2
        shift 2
        ;;
    -*|--*)
      echo "Unknown option $1"
      shift
      ;;
  esac
done

SCRIPTDIR=`dirname "$0"`
SCRIPTDIR=`realpath ${SCRIPTDIR}`

CURDIR=`pwd`
CURDIR=`realpath ${CURDIR}`

LANDOPT_HOME=${SCRIPTDIR}/landscape-optimization

SCRATCH=${CURDIR}/scratch
INPUTS=${CURDIR}/inputs
OUTPUTS=${CURDIR}/outputs

FULL_IGNITIONS_CSV=${INPUTS}/full_ignitions.csv
RX_BURN_UNITS_DIR=${INPUTS}/rx_burn_units
BURNED_AREA_DIR=${INPUTS}/burned_area
BUILDING_DAMAGE_DIR=${INPUTS}/building_damage
HABITAT_DAMAGE_DIR=${INPUTS}/habitat_damage
INTENSITY_DIR=${INPUTS}/intensity

RESULTS_DIR=${SCRATCH}/run
CONFIG_FILE=${SCRATCH}/config.yaml

OUTPUT_PREVENTION_TABLE=${OUTPUTS}/prevention_table.csv
OUTPUT_VALUES_TABLE=${OUTPUTS}/values_table.csv
OUTPUT_INITIAL_HAZARD_FILE=${OUTPUTS}/initial_hazard.tif
OUTPUT_MERGED_HAZARD_FILE=${OUTPUTS}/merged_hazard.tif

rm -f -r $SCRATCH $INPUTS $OUTPUTS

mkdir $SCRATCH $INPUTS $OUTPUTS
mkdir $RX_BURN_UNITS_DIR $BURNED_AREA_DIR $BUILDING_DAMAGE_DIR 
mkdir $HABITAT_DAMAGE_DIR $INTENSITY_DIR $RESULTS_DIR

cp $FULL_IGNITIONS_FILE $FULL_IGNITIONS_CSV
tar -xzf $RX_BURN_UNITS_FILE -C $RX_BURN_UNITS_DIR >/dev/null 2>&1
tar -xzf $BURNED_AREA_FILE -C $BURNED_AREA_DIR >/dev/null 2>&1
tar -xzf $BUILDING_DAMAGE_FILE -C $BUILDING_DAMAGE_DIR >/dev/null 2>&1
tar -xzf $HABITAT_DAMAGE_FILE -C $HABITAT_DAMAGE_DIR >/dev/null 2>&1
tar -xzf $INTENSITY_FILE -C $INTENSITY_DIR >/dev/null 2>&1

{
  echo "alpha: 0.1"
  echo "budget: 10"
  echo "full_ignitions_file_path: $FULL_IGNITIONS_CSV"  
  echo "rx_burn_units_path: $RX_BURN_UNITS_DIR"  
  echo "burned_area_dir: $BURNED_AREA_DIR"
  echo "bldg_dmg_dir: $BUILDING_DAMAGE_DIR"  
  echo "habitat_dmg_dir: $HABITAT_DAMAGE_DIR"
  echo "intensity_dir: $INTENSITY_DIR"
  echo "results_dir: $RESULTS_DIR"
  echo "prevention_file_path: $OUTPUT_PREVENTION_TABLE"
  echo "values_file_path: $OUTPUT_VALUES_TABLE"
} >> $CONFIG_FILE

cd $SCRATCH

# Preprocess 1
python3 ${LANDOPT_HOME}/preprocess_p1.py

# Run Hazard Subchunks Handler
python3 ${LANDOPT_HOME}/hazard/create_subchunks.py

## Create bounds
cd $INTENSITY_DIR
ls | grep tif | xargs -n1000 gdaltindex bounds.shp
cd $SCRATCH

## Start Dask
dask scheduler --port 8786 2>&1 >/dev/null &
spid=$!
sleep 5
dask worker --nthreads 1 --nworkers 2 --memory-limit 1.8GB tcp://localhost:8786 2>&1 >/dev/null &
wpid=$!
sleep 5

## Preprocess hazard subtrunks 
ulimit -n 100000
find ${RESULTS_DIR}/hazard_config/ -type f | sort | xargs python3 ${LANDOPT_HOME}/hazard/preprocess_hazard_subtrunks.py -s tcp://localhost:8786 -c
sleep 5

# Stop Dask
kill $spid $wpid

# Merge subchunks
python3 ${LANDOPT_HOME}/hazard/gdal_merge.py -o ${RESULTS_DIR}/merged.tif ${RESULTS_DIR}/hazards/subchunks*.tif

# Preprocess 2 (Update )
python3 ${LANDOPT_HOME}/preprocess_p2.py

cp ${RESULTS_DIR}/initial_Hazard.tif $OUTPUT_INITIAL_HAZARD_FILE
cp ${RESULTS_DIR}/merged.tif $OUTPUT_MERGED_HAZARD_FILE

cd $CURDIR

# Clean up and exit:
#rm -f -r $SCRATCH

exit 0
