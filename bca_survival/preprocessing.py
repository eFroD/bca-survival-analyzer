"""
Survival Data Preprocessing Module.

This module provides utility functions for preprocessing survival analysis data, including
calculating time-to-event durations, handling missing or invalid data, creating event indicators,
and computing tissue ratios from the BCA values.

Requires: pandas
"""

import pandas as pd


def calculate_days(df, start_date_col, event_date_col, event_col):
    """
    Calculates the number of days between two date columns and sets an event indicator.

    Args:
        df (pd.DataFrame): The input dataframe.
        start_date_col (str): Name of the column containing start dates.
        event_date_col (str, optional): Name of the column containing event dates.
            If None, only the event indicator will be created.
        event_col (str): Name of the column containing event indicators (1/0 or True/False).

    Returns:
        pd.DataFrame: DataFrame with added 'days' and 'event' columns.

    Note:
        The function expects dates in the format '%d.%m.%Y' (e.g., '31.12.2020').
        The 'days' column represents the time between start and event dates.
        The 'event' column is converted to integer type.
    """
    if event_date_col:
        df["days"] = (
            pd.to_datetime(df[event_date_col], format="%d.%m.%Y")
            - pd.to_datetime(df[start_date_col], format="%d.%m.%Y")
        ).dt.days
    df["event"] = df[event_col].astype(int)

    return df


def check_and_remove_negative_days(df):
    """
    Checks for and removes rows with negative or NaN values in the 'days' column.

    Args:
        df (pd.DataFrame): The input dataframe with a 'days' column.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: DataFrame with negative and NaN 'days' values removed.
            - pd.DataFrame or None: DataFrame containing only the removed rows, or None if no rows were removed.

    Note:
        Negative days values can occur due to data entry errors or when an event occurs before
        the recorded start date. This function identifies and removes such problematic records.
        It prints a warning if any rows are removed.
    """
    negative_values_count = (df["days"] < 0).sum()
    nan_count = (df["days"].isna()).sum()
    df_negative = None
    if negative_values_count > 0:
        print(
            f"Warning: {negative_values_count + nan_count} rows will be dropped because they contain values below 0 or nan in the 'days' column."
        )
        df_negative = df[(df["days"] < 0) | (df["days"].isna())]
        df = df[df["days"] >= 0]

    return df, df_negative


def create_event_date_column(df, date_death, date_disease_death, date_followup):
    """
    Creates an event date column and event indicator based on multiple date columns.
    This is used to prepare for Overall Survival analysis.

    Args:
        df (pd.DataFrame): The input dataframe.
        date_death (str): Column name containing the date of death.
        date_disease_death (str): Column name containing the date of disease-specific death.
        date_followup (str): Column name containing the date of last follow-up.

    Returns:
        pd.DataFrame: DataFrame with added 'event_date' and 'event' columns.

    Note:
        This function prioritizes death dates over follow-up dates. It sets the event
        indicator to True if either death date is present, and False if only the follow-up
        date is available. If no dates are available, both columns are set to NaN.
    """
    for i, row in df.iterrows():
        if not pd.isna(row[date_death]):
            df.loc[i, "event_date"] = row[date_death]
            df.loc[i, "event"] = True
        elif not pd.isna(row[date_disease_death]):
            df.loc[i, "event_date"] = row[date_disease_death]
            df.loc[i, "event"] = True
        elif not pd.isna(row[date_followup]):
            df.loc[i, "event_date"] = row[date_followup]
            df.loc[i, "event"] = False
        else:
            df.loc[i, "event_date"] = pd.NA
            df.loc[i, "event"] = pd.NA
    return df


def compute_ratios(df):
    """
    Computes ratios between different tissue measurements across body parts and metrics.

    Args:
        df (pd.DataFrame): The input dataframe containing tissue measurement columns.

    Returns:
        pd.DataFrame: DataFrame with additional columns for computed ratios.

    Note:
        This function calculates ratios such as intramuscular adipose tissue to total adipose tissue
        (imat/tat), visceral fat to total fat (vat/tat), etc., for various body parts and metrics.

        The column naming convention is:
        '{body_part}::WL::{tissue_type}::{metric}' for measurements
        '{body_part}::WL::{numerator}/{denominator}::{metric}' for ratios

        For example, 'l5::WL::imat/tat::mean_ml' represents the ratio of mean milliliter
        volume of intramuscular adipose tissue to total adipose tissue at the L5 vertebra level.
    """
    # Define the parts and metrics
    body_parts = [
        "ventral_cavity",
        "abdominal_cavity",
        "thoracic_cavity",
        "mediastinum",
        "pericardium",
        "l5",
        "l4",
        "l3",
        "l2",
        "l1",
        "t12",
        "t11",
        "t10",
        "t9",
        "t8",
        "t7",
        "t6",
        "t5",
        "t4",
        "t3",
        "t2",
        "t1",
    ]
    metrics = [
        "mean_ml",
        "std_ml",
        "min_ml",
        "q1_ml",
        "q2_ml",
        "q3_ml",
        "max_ml",
        "sum_ml",
        "mean_hu",
    ]

    ratios = [
        ("imat", "tat"),
        ("vat", "tat"),
        ("eat", "tat"),
        ("sat", "tat"),
        ("pat", "tat"),
        ("muscle", "bone"),
        ("imat", "muscle"),
    ]

    # Iterate through each body part and metric combination
    for body_part in body_parts:
        for metric in metrics:
            for numerator, denominator in ratios:
                num_col = f"{body_part}::WL::{numerator}::{metric}"
                den_col = f"{body_part}::WL::{denominator}::{metric}"
                new_col = f"{body_part}::WL::{numerator}/{denominator}::{metric}"

                if num_col in df.columns and den_col in df.columns:
                    df[new_col] = df[num_col] / df[den_col]

    return df
