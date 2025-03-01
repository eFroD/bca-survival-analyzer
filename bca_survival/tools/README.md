# BOA Results Processor
## Overview
This script processes BOA results data by extracting and compiling measurements from JSON files across the individual BOA resutls. It specifically targets two types of measurement files:
- `total-measurements.json` - Contains segmentation measurements for the organ segmentation
- `bca-measurements.json` - Contains body composition analysis measurements

The script traverses a directory structure, processes these files, and outputs consolidated Excel spreadsheets containing all the measurement data.

## Purpose
The primary purpose is to aggregate measurement data from multiple patient studies into standardized datasets for further analysis. This is the first step that needs to be taken prior to the actual Cox regression using BCA values.

## Requirements
- Python 3.9 or higher
- Required libraries:
  - os
  - json
  - pandas
  - argparse
  - pathlib

## Usage
```bash
python script_name.py base_path output_path
```

### Arguments
- `base_path`: The root directory containing the folder structure with JSON measurement files
- `output_path`: Directory where the resulting Excel files will be saved

## Folder Structure Assumptions
The script assumes a specific directory structure:
```
root_directory/
├── study_id_1/
│   ├── scan_folder/
│   │   ├── total-measurements.json
│   │   └── bca-measurements.json
├── study_id_2/
│   ├── scan_folder/
│   │   ├── total-measurements.json
│   │   └── bca-measurements.json
...
```

Where:
- Each study has a unique ID as its folder name
- Each study folder contains data with the measurement JSON files

## Functions

### `main(root_path, output_path)`
The entry point that orchestrates the data processing and file saving operations.

### `process_json_files(root_dir)`
Walks through the directory structure, identifies relevant JSON files, processes them, and compiles the data into pandas DataFrames.

Returns:
- `final_total_df`: DataFrame containing consolidated organ segmentation measurements
- `final_bca_df`: DataFrame containing consolidated body composition analysis measurements

### `process_totalseg_measurements(file_path, dirpath)`
Processes an individual `total-measurements.json` file to extract organ segmentation measurements.

Parameters:
- `file_path`: Path to the JSON file
- `dirpath`: Directory path containing the file (used to extract the study ID)

Returns:
- A pandas DataFrame with one row representing the measurements from the file

### `process_bca_measurements(folder_path)`
Processes an individual `bca-measurements.json` file to extract body composition analysis measurements.

Parameters:
- `folder_path`: Path to the folder containing the JSON file

Returns:
- A pandas DataFrame with one row representing the measurements from the file, or None if the file doesn't exist

## Output
The script generates two Excel files in the specified output directory:
1. `total-measurements.xlsx`: Contains all organ segmentation measurements
2. `bca-measurements.xlsx`: Contains all body composition analysis measurements

## Data Structure
- All measurements are organized by study ID
- Column names follow the pattern:
  - For total measurements: `organ::metric`
  - For BCA measurements: `scan_key::type::key::metric_unit`

## Notes
- The script skips folders that don't contain the required JSON files
- Progress is reported to the console as each file is processed
- If no files of a certain type are found, an empty DataFrame is created