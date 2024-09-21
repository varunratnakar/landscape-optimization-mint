arguments:
- --
baseCommand: /landopt/run.sh
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
  values_table_file:
    inputBinding:
      prefix: --values_table_file
    type: File
  prevention_table_file:
    inputBinding:
      prefix: --prevention_table_file
    type: File  
  initial_hazard_file:
    inputBinding:
      prefix: --initial_hazard_file
    type: File  
  merged_hazard_file:
    inputBinding:
      prefix: --merged_hazard_file
    type: File          
  budget:
    inputBinding:
      prefix: --budget
    type: int
outputs:
  solutions_file:
    outputBinding:
      glob: ./outputs/solutions.csv
    type: File
  solutions_values_file:
    outputBinding:
      glob: ./outputs/solutions_values.csv
    type: File
  heatmaps:
    outputBinding:
      glob: ./outputs/heatmaps.tar.gz
    type: File    
requirements:
  NetworkAccess:
    networkAccess: true
