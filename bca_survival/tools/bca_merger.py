#!/usr/bin/env python3
"""
bca_merger - A CLI tool to merge two Excel files based on ID columns.

Usage:
    bca_merger <first_file> <second_file> <id_column_name>

Arguments:
    first_file      Path to the first Excel file
    second_file     Path to the second Excel file
    id_column_name  Name of the ID column in the first file to match with 'StudyID' in the second file
"""

import sys
from pathlib import Path
from typing import Union

import pandas as pd


def merge_files(
    first_file_path: Union[str, Path], second_file_path: Union[str, Path], id_column_name: str
) -> bool:
    """
    Merge two Excel files based on ID columns.

    Args:
        first_file_path: Path to the first Excel file
        second_file_path: Path to the second Excel file
        id_column_name: Name of the ID column in the first file

    Returns:
        bool: True if merge was successful, False otherwise
    """
    try:
        # Read the Excel files
        # Use 'openpyxl' engine and parse_dates=False to keep date columns as they are
        # Set index_col=None explicitly to avoid creating an index column
        df1: pd.DataFrame = pd.read_excel(
            first_file_path, engine="openpyxl", parse_dates=False, index_col=None
        )
        df2: pd.DataFrame = pd.read_excel(
            second_file_path, engine="openpyxl", parse_dates=False, index_col=None
        )

        # Verify that the ID column exists in the first file
        if id_column_name not in df1.columns:
            print(f"Error: Column '{id_column_name}' not found in the first file.")
            sys.exit(1)

        # Verify that 'StudyID' column exists in the second file
        if "StudyID" not in df2.columns:
            print("Error: Column 'StudyID' not found in the second file.")
            sys.exit(1)

        # Merge the dataframes
        # Using outer merge to keep all rows from both files
        # and fill with NaN where there's no match
        merged_df: pd.DataFrame = pd.merge(
            df1, df2, left_on=id_column_name, right_on="StudyID", how="outer", indicator=False
        )

        # Remove the duplicate StudyID column if it exists and is different from the id_column_name
        if "StudyID" in merged_df.columns and "StudyID" != id_column_name:
            merged_df = merged_df.drop(columns=["StudyID"])

        # Generate output filename
        first_file_name: str = Path(first_file_path).stem
        output_path: str = f"{first_file_name}_merged.xlsx"

        # Convert datetime columns to date only (remove time component)
        for col in merged_df.columns:
            # Check if the column contains datetime values
            if pd.api.types.is_datetime64_any_dtype(merged_df[col]):
                # Convert to date only (removes time component)
                merged_df[col] = merged_df[col].dt.strftime("%d.%m.%Y")

        # Write the merged dataframe to a new Excel file
        # Make sure index=False to prevent unnamed columns
        with pd.ExcelWriter(output_path, engine="openpyxl", date_format="%d.%m.%Y") as writer:
            merged_df.to_excel(writer, index=False)

        print(f"Merged file created: {output_path}")
        return True

    except Exception as e:
        print(f"Error: {str(e)}")
        return False


def main() -> None:
    """
    Main function to parse command line arguments and execute the merge.
    """
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)

    first_file: str = sys.argv[1]
    second_file: str = sys.argv[2]
    id_column_name: str = sys.argv[3]

    # Check if files exist
    if not Path(first_file).is_file():
        print(f"Error: First file '{first_file}' does not exist.")
        sys.exit(1)

    if not Path(second_file).is_file():
        print(f"Error: Second file '{second_file}' does not exist.")
        sys.exit(1)

    # Merge the files
    print("Files are ok. Beginning merging...")
    success: bool = merge_files(first_file, second_file, id_column_name)

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
