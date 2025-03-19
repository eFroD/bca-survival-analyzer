"""
BOA Results extractor

This script processes BOA data by extracting measurements from the individual JSON files
across a directory structure. It targets two types of measurement files:
- total-measurements.json: Contains segmentation measurements for various organs
- bca-measurements.json: Contains body composition analysis measurements

The script consolidates these measurements into Excel spreadsheets for further analysis.

Usage:
    boa-extract base_path output_path
    python -m survival_analysis.boa_extractor base_path output_path

Author: Eric
"""

import argparse
import json
import os
from pathlib import Path
from typing import Tuple, Union

import pandas as pd


def process_json_files(root_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Walks through the directory structure, identifies relevant JSON files,
    processes them, and compiles the data into pandas DataFrames.

    Args:
        root_dir (str): The root directory to search for measurement files

    Returns:
        tuple: A tuple containing two DataFrames:
            - final_total_df: DataFrame with organ segmentation measurements
            - final_bca_df: DataFrame with body composition analysis measurements
    """
    totalseg_data = []
    bca_data = []
    file_count = 0
    root_dir = str(Path(root_dir))
    print("Evaluating all files in {}".format(root_dir))
    # Walk through directory
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if "total-measurements.json" in filenames:
            file_path = os.path.join(dirpath, "total-measurements.json")
            totalseg_data.append(process_totalseg_measurements(file_path, dirpath))
            file_count += 1
            bca_df = process_bca_measurements(dirpath)
            if bca_df is not None:
                bca_data.append(bca_df)

            # Report progress
            print(f"Processed: {file_path}")

    # Create DataFrame
    if totalseg_data:
        final_total_df = pd.concat(totalseg_data, ignore_index=True)
    else:
        print("No total-measurements files found.")
        final_total_df = pd.DataFrame()

    # Concatenate and save bca measurements data to CSV
    if bca_data:
        final_bca_df = pd.concat(bca_data, ignore_index=True)
    else:
        print("No Processable bca files found")
        final_bca_df = pd.DataFrame()

    return final_total_df, final_bca_df


def process_totalseg_measurements(file_path: str, dirpath: str) -> pd.DataFrame:
    """
    Processes an individual total-measurements.json file to extract organ segmentation measurements.

    Args:
        file_path (str): Path to the JSON file
        dirpath (str): Directory path containing the file (used to extract the study ID)

    Returns:
        pandas.DataFrame: A DataFrame with one row representing the measurements from the file
    """
    with open(file_path, "r") as file:
        content = json.load(file)

        # Get the parent folder name
        study_id = os.path.basename(os.path.dirname(dirpath))

        # Prepare the data for the dataframe
        row_data = {"StudyID": study_id}

        # Iterate through the 'total' key
        if "segmentations" in content and "total" in content["segmentations"]:
            total_data = content["segmentations"]["total"]
            for organ, metrics in total_data.items():
                if metrics.get("present"):
                    for metric, value in metrics.items():
                        if metric != "present":
                            column_name = f"{organ}::{metric}"
                            row_data[column_name] = value
        return pd.DataFrame([row_data])


def process_bca_measurements(folder_path: str) -> Union[pd.DataFrame, None]:
    """
    Processes the bca-measurements.json file in the given folder and extracts the measurement data.

    Args:
        folder_path (str): The path to the folder containing the bca-measurements.json file.

    Returns:
        pandas.DataFrame or None: A DataFrame containing the measurement data with formatted
                                column names, or None if the file doesn't exist.
    """
    file_path = os.path.join(folder_path, "bca-measurements.json")
    if not os.path.exists(file_path):
        return None

    with open(file_path, "r") as file:
        data = json.load(file)

    aggregated = data.get("aggregated", {})
    study_id = os.path.basename(os.path.dirname(folder_path))
    row = {"StudyID": study_id}

    for scan_key, scan_value in aggregated.items():
        for measurement_type, measurements in scan_value.items():
            if isinstance(measurements, int):
                continue
            prefix = f"{scan_key}::{'ALL' if measurement_type == 'measurements' else 'WL'}::"
            for key, value in measurements.items():
                for metric, value in value.items():
                    column_name = (
                        f"{prefix}{key}::{metric if metric.endswith('_hu') else metric + '_ml'}"
                    )
                    row[column_name] = value

    df = pd.DataFrame([row])
    return df


def main(root_path: str, output_path: str) -> None:
    """
    Main function to iterate over folders, process the JSON files, and save the results to Excel files.

    Args:
        root_path (str): The base directory containing the folders with JSON files
        output_path (str): Path to save the resulting Excel files
    """
    total_df, bca_df = process_json_files(root_path)
    if "_" in total_df["StudyID"][0]:
        bca_df["StudyID"] = bca_df["StudyID"].apply(lambda x: x.split("_")[1]).astype(int)
        total_df["StudyID"] = total_df["StudyID"].apply(lambda x: x.split("_")[1]).astype(int)
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Save results to Excel files
    total_df.to_excel(os.path.join(output_path, "total-measurements.xlsx"), index=False)
    bca_df.to_excel(os.path.join(output_path, "bca-measurements.xlsx"), index=False)

    print(f"Results saved to {output_path}")
    print(
        f"Found {len(total_df)} total measurement records and {len(bca_df)} BCA measurement records"
    )


def main_cli() -> None:
    """
    Command-line interface entry point for the BOA extractor.
    This function is referenced in pyproject.toml to create the console script.
    """
    parser = argparse.ArgumentParser(
        description="Process BOA JSON files and export measurements to Excel files."
    )
    parser.add_argument(
        "base_path", type=str, help="The base directory containing the folders with JSON files."
    )
    parser.add_argument("output_path", type=str, help="Path to save the resulting Excel files")
    #    parser.add_argument('--version', action='version', version=f'%(prog)s {get_version()}')
    args = parser.parse_args()

    main(args.base_path, args.output_path)


# def get_version() -> str:
#    """
#    Get the version of the package.#

#    Returns:
#        str: The version string or 'development' if not available
#    """
#    try:
#        from bca_survival._version import version
#        return version
#    except ImportError:
#        return "development"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process BOA JSON files and export measurements to Excel files."
    )
    parser.add_argument(
        "base_path", type=str, help="The base directory containing the folders with JSON files."
    )
    parser.add_argument("output_path", type=str, help="Path to save the resulting Excel files")
    # parser.add_argument('--version', action='version', version=f'%(prog)s {get_version()}')
    args = parser.parse_args()

    main(args.base_path, args.output_path)
