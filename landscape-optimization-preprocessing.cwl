arguments:
- --
baseCommand: /landopt/preprocess.sh
class: CommandLineTool
cwlVersion: v1.1
hints:
  DockerRequirement:
    dockerImageId: kcapd/landscape-optimization-mint:latest
inputs:
  full_ignitions_file:
    inputBinding:
      prefix: --full_ignitions_file
    type: File
  rx_burn_units_file:
    inputBinding:
      prefix: --rx_burn_units_file
    type: File
  burned_area_file:
    inputBinding:
      prefix: --burned_area_file
    type: File
  building_damage_file:
    inputBinding:
      prefix: --building_damage_file
    type: File
  habitat_damage_file:
    inputBinding:
      prefix: --habitat_damage_file
    type: File
  intensity_file:
    inputBinding:
      prefix: --intensity_file
    type: File
outputs:
  values_table_file:
    outputBinding:
      glob: ./outputs/values_table.csv
    type: File
  prevention_table_file:
    outputBinding:
      glob: ./outputs/prevention_table.csv
    type: File
  initial_hazard_file:
    outputBinding:
      glob: ./outputs/initial_hazard.tif
    type: File
  merged_hazard_file:
    outputBinding:
      glob: ./outputs/merged_hazard.tif
    type: File      
requirements:
  NetworkAccess:
    networkAccess: true
