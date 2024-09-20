#!/bin/bash
set -xe
FULL_IGNITIONS_FILE=""
RX_BURN_UNITS_FILE=""
BURNED_AREA_FILE=""
BUILDING_DAMAGE_FILE=""
HABITAT_DAMAGE_FILE=""
INTENSITY_FILE=""
VALUES_TABLE_FILE=""
PREVENTION_TABLE_FILE=""
INITIAL_HAZARD_FILE=""
MERGED_HAZARD_FILE=""

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
    -v | --values_table_file)
        echo "Processing 'values_table_file' option. Input argument is '$2'"
        VALUES_TABLE_FILE=$2
        shift 2
        ;;
    -p | --prevention_table_file)
        echo "Processing 'prevention_table_file' option. Input argument is '$2'"
        PREVENTION_TABLE_FILE=$2
        shift 2
        ;;
    -z | --initial_hazard_file)
        echo "Processing 'initial_hazard_file' option. Input argument is '$2'"
        INITIAL_HAZARD_FILE=$2
        shift 2
        ;;
    -m | --merged_hazard_file)
        echo "Processing 'merged_hazard_file' option. Input argument is '$2'"
        MERGED_HAZARD_FILE=$2
        shift 2
        ;;
    -a | --alpha)
        echo "Processing 'alpha' option. Input argument is '$2'"
        ALPHA=$2
        shift 2
        ;;
    -b | --budget)
        echo "Processing 'budget' option. Input argument is '$2'"
        BUDGET=$2
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
VALUES_TABLE_CSV=${INPUTS}/values_table.csv
PREVENTION_TABLE_CSV=${INPUTS}/prevention_table.csv

INITIAL_HAZARD_TIF=${SCRATCH}/initial_Hazard.tif
MERGE_HAZARD_TIF=${SCRATCH}/merged.tif

RX_BURN_UNITS_DIR=${INPUTS}/rx_burn_units
BURNED_AREA_DIR=${INPUTS}/burned_area
BUILDING_DAMAGE_DIR=${INPUTS}/building_damage
HABITAT_DAMAGE_DIR=${INPUTS}/habitat_damage
INTENSITY_DIR=${INPUTS}/intensity

RESULTS_DIR=${SCRATCH}/run
CONFIG_FILE=${SCRATCH}/config.yaml

OUTPUT_PREVENTION_TABLE=${OUTPUTS}/prevention_table.csv
OUTPUT_VALUES_TABLE=${OUTPUTS}/values_table.csv

rm -f -r $SCRATCH $INPUTS $OUTPUTS

mkdir $SCRATCH $INPUTS $OUTPUTS
mkdir $RX_BURN_UNITS_DIR $BURNED_AREA_DIR $BUILDING_DAMAGE_DIR
mkdir $HABITAT_DAMAGE_DIR $INTENSITY_DIR $RESULTS_DIR

cp $FULL_IGNITIONS_FILE $FULL_IGNITIONS_CSV
cp $VALUES_TABLE_FILE $VALUES_TABLE_CSV
cp $PREVENTION_TABLE_FILE $PREVENTION_TABLE_CSV
cp $INITIAL_HAZARD_FILE $INITIAL_HAZARD_TIF
cp $MERGED_HAZARD_FILE $MERGE_HAZARD_TIF

tar -xzf $RX_BURN_UNITS_FILE -C $RX_BURN_UNITS_DIR >/dev/null 2>&1
tar -xzf $BURNED_AREA_FILE -C $BURNED_AREA_DIR >/dev/null 2>&1
tar -xzf $BUILDING_DAMAGE_FILE -C $BUILDING_DAMAGE_DIR >/dev/null 2>&1
tar -xzf $HABITAT_DAMAGE_FILE -C $HABITAT_DAMAGE_DIR >/dev/null 2>&1
tar -xzf $INTENSITY_FILE -C $INTENSITY_DIR >/dev/null 2>&1

{
  echo "alpha: $ALPHA"
  echo "budget: $BUDGET"
  echo "full_ignitions_file_path: $FULL_IGNITIONS_CSV"
  echo "rx_burn_units_path: $RX_BURN_UNITS_DIR"
  echo "burned_area_dir: $BURNED_AREA_DIR"
  echo "bldg_dmg_dir: $BUILDING_DAMAGE_DIR"
  echo "habitat_dmg_dir: $HABITAT_DAMAGE_DIR"
  echo "intensity_dir: $INTENSITY_DIR"
  echo "results_dir: $RESULTS_DIR"
  echo "prevention_file_path: $PREVENTION_TABLE_CSV"
  echo "values_file_path: $VALUES_TABLE_CSV"
} >> $CONFIG_FILE

cd $SCRATCH

# Main Landscape optimization
python3 ${LANDOPT_HOME}/main.py
cp ${RESULTS_DIR}/solutions.csv ${OUTPUTS}/solutions.csv
cp ${RESULTS_DIR}/solutions_values.csv ${OUTPUTS}/solutions_values.csv

# Heatmaps generation
python3 ${LANDOPT_HOME}/heatmap.py
cd ${RESULTS_DIR}/heatmaps
tar -cvzf ${OUTPUTS}/heatmaps.tar.gz *.tif

cd $CURDIR

# Clean up and exit:
#rm -f -r $SCRATCH

exit 0
