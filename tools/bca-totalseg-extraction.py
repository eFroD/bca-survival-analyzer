import os
import json
import pandas as pd
import argparse
from pathlib import Path


def process_json_files(root_dir):
    totalseg_data = []
    bca_data = []
    file_count = 0
    root_dir = str(Path(root_dir))
    # Walk through directory
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if 'total-measurements.json' in filenames:
            file_path = os.path.join(dirpath, 'total-measurements.json')
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


        # Concatenate and save bca measurements data to CSV
    if bca_data:
        final_bca_df = pd.concat(bca_data, ignore_index=True)

    #combined_df = pd.merge(final_bca_df, final_total_df, on="StudyID", how="left")
    return final_total_df, final_bca_df


def process_totalseg_measurements(file_path, dirpath):
    with open(file_path, 'r') as file:
        content = json.load(file)

        # Get the parent folder name
        study_id = os.path.basename(os.path.dirname(dirpath))

        # Prepare the data for the dataframe
        row_data = {'StudyID': study_id}

        # Iterate through the 'total' key
        if 'segmentations' in content and 'total' in content['segmentations']:
            total_data = content['segmentations']['total']
            for organ, metrics in total_data.items():
                if metrics.get('present'):
                    for metric, value in metrics.items():
                        if metric != 'present':
                            column_name = f"{organ}::{metric}"
                            row_data[column_name] = value
        return pd.DataFrame([row_data])


def process_bca_measurements(folder_path):
    """
    Processes the bca-measurements.json file in the given folder and extracts the measurement data.

    Args:
        folder_path (str): The path to the folder containing the bca-measurements.json file.

    Returns:
        pandas.DataFrame: A DataFrame containing the measurement data with formatted column names.
    """
    file_path = os.path.join(folder_path, 'bca-measurements.json')
    if not os.path.exists(file_path):
        return None

    with open(file_path, 'r') as file:
        data = json.load(file)

    aggregated = data.get('aggregated', {})
    study_id = os.path.basename(os.path.dirname(folder_path))
    row = {'StudyID': study_id}

    for scan_key, scan_value in aggregated.items():
        for measurement_type, measurements in scan_value.items():
            if isinstance(measurements, int):
                continue
            prefix = f"{scan_key}::{'ALL' if measurement_type == 'measurements' else 'WL'}::"
            for key, value in measurements.items():
                for metric, value in value.items():
                    column_name = f"{prefix}{key}::{metric if metric.endswith('_hu') else metric + '_ml'}"
                    row[column_name] = value

    df = pd.DataFrame([row])
    return df


def main(root_path, output_path):
    """
    Main function to iterate over folders, process the JSON files, and save the results to CSV files.
    """
    total_df, bca_df = process_json_files(root_path)
    total_df.to_excel(os.path.join(output_path, 'total-measurements.xlsx'), index=False)
    bca_df.to_excel(os.path.join(output_path, 'bca-measurements.xlsx'), index=False)
    #combined_df.to_csv(os.path.join(root_path, str(output_path)+'bca_and_totalseg_measurements.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process JSON files and merge data into a single CSV file.')
    parser.add_argument('base_path', type=str, help='The base directory containing the folders with JSON files.')
    parser.add_argument('output_path', type=str, help='Path to save the resulting DataFrame as a CSV file')
    args = parser.parse_args()
    main(args.base_path, args.output_path)

