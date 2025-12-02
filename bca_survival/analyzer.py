"""
BCA Survival Analyzer Module.

This module provides a class for performing survival analysis on body composition
assessment (BCA) data. It combines clinical/demographic data with body measurement data
and provides methods for univariate and multivariate Cox regression analysis, as well as
Kaplan-Meier survival curves.

Requires: pandas, numpy, and custom preprocessing and models modules
"""

from typing import List, Optional, Tuple, Union

import lifelines
import numpy as np
import pandas as pd

from .models import (
    generate_kaplan_meier_plot,
    perform_multivariate_cox_regression,
    perform_univariate_cox_regression,
)
from .preprocessing import calculate_days, check_and_remove_negative_days


class BCASurvivalAnalyzer:
    """
    A class for analyzing the relationship between body composition measurements and survival outcomes.

    This class combines clinical/demographic data with body composition measurements,
    preprocesses the data for survival analysis, and provides methods for performing
    Cox regression and Kaplan-Meier survival analysis.

    Attributes:
        df (pd.DataFrame): The merged and preprocessed dataframe.
        df_negative_days (pd.DataFrame): Dataframe containing records with negative or NaN 'days' values for evaluation.
        start_date_col (str): Name of the column containing start dates.
        event_date_col (str): Name of the column containing event dates.
        event_col (str): Name of the column containing event indicators.
        standardize (bool): Whether to standardize variables for analysis.
    """

    def __init__(
        self,
        df_main: pd.DataFrame,
        df_measurements: pd.DataFrame,
        main_id_col: str,
        measurement_id_col: str,
        start_date_col: str,
        event_date_col: str,
        event_col: str,
        standardize: bool = False,
    ):
        """
        Initializes the BCASurvivalAnalyzer with clinical and measurement data.

        Args:
            df_main (pd.DataFrame): Dataframe containing clinical/demographic data.
            df_measurements (pd.DataFrame): Dataframe containing body composition measurements.
            main_id_col (str): Name of the PID column in df_main.
            measurement_id_col (str): Name of the PID column in df_measurements.
            start_date_col (str): Name of the column containing start dates.
            event_date_col (str): Name of the column containing event dates.
            event_col (str): Name of the column containing event indicators.
            standardize (bool, optional): Whether to standardize variables for analysis. Defaults to False.

        Note:
            The function renames ID columns to 'PID' for consistency, replaces 'nd' with NaN,
            and handles infinite values. It also checks for and warns about missing measurements.
        """
        # Rename ID columns to unify them across both dataframes
        df_main = df_main.rename(columns={main_id_col: "PID"})
        df_measurements = df_measurements.rename(columns={measurement_id_col: "PID"})
        df_main.replace("nd", np.nan, inplace=True)
        df_measurements.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Merge the main and measurements dataframes on the unified ID column
        len_main = len(df_main)
        len_measurements = len(df_measurements)
        if len_main != len_measurements:
            missing_pids = df_main[~df_main["PID"].isin(df_measurements["PID"])]

            if not missing_pids.empty:
                print("Warning: The following PIDs have no BCA values:")
                print(missing_pids["PID"].unique())
                df_main = df_main[df_main["PID"].isin(df_measurements["PID"])]

        self.df = pd.merge(df_main, df_measurements, on="PID", how="left")
        self.df_negative_days = None
        self.start_date_col = start_date_col
        self.event_date_col = event_date_col
        self.event_col = event_col
        self.standardize = standardize
        self.preprocess_data()

    def preprocess_data(self) -> pd.DataFrame:
        """
        Preprocesses the data for survival analysis.

        This method calculates time-to-event days and removes records with negative days.

        Returns:
            pd.DataFrame: The preprocessed dataframe.

        Note:
            This method is called automatically during initialization but can be
            called again if the underlying data changes.
        """
        self.df = calculate_days(self.df, self.start_date_col, self.event_date_col, self.event_col)
        self.df, self.df_negative_days = check_and_remove_negative_days(self.df)
        return self.df

    def univariate_cox_regression(
        self,
        columns: List[str],
        verbose: bool = False,
        penalizer: float = 0.0,
        correction_values: Union[List[str], None] = None,
        nan_threshold: float = 0.7,
        significant_only: bool = True,
    ) -> pd.DataFrame:
        """
        Performs univariate Cox proportional hazards regression for each specified variable.

        Args:
            columns (list): List of predictor column names to test individually.
            verbose (bool, optional): Whether to print detailed progress information. Defaults to False.
            penalizer (float, optional): L2 penalizer value to apply to the regression. Defaults to 0.0.
            correction_values (list, optional): List of column names to include as correction terms in each
                univariate model. Defaults to None.
            nan_threshold (float, optional): Threshold for NaN values if standardizing. Defaults to 0.7.
            significant_only (bool, optional): Whether to only include significant observations. Defaults to True.

        Returns:
            pd.DataFrame: DataFrame containing significant variables and their statistics.

        Note:
            This method tests each variable individually in a Cox regression model,
            and returns only statistically significant variables (p < 0.05).
            If correction_values is provided, those variables will be included in each model
            to adjust for their effects.
        """
        if correction_values is None:
            correction_values = []
        return perform_univariate_cox_regression(
            self.df,
            columns.copy(),
            self.standardize,
            verbose=verbose,
            penalizer=penalizer,
            correction_values=correction_values,
            nan_threshold=nan_threshold,
            significant_only=significant_only,
        )

    def kaplan_meier_plot(
        self,
        column: str,
        split_strategy: str = "median",
        fixed_value: Union[float, None] = None,
        output_path: Union[str, None] = None,
        percentage: Union[float, None] = None,
        custom_title: Optional[str] = None,
        dpi: int = 400,
        custom_high_low_names: Tuple[str, str] = ("low", "high"),
    ) -> dict:
        """
        Generates a Kaplan-Meier survival plot for a specified variable.

        Args:
            column (str): Column name to use for grouping.
            split_strategy (str, optional): Strategy for splitting data into high/low groups.
                Options: 'mean', 'median', 'percentage', 'fixed'. Defaults to 'median'.
            fixed_value (float, optional): Fixed threshold value when split_strategy is 'fixed'.
                Defaults to None.
            output_path (str, optional): Directory path to save the plot. If None, saves in current
                directory. Defaults to None.
            percentage (float, optional): Percentile threshold when split_strategy is 'percentage'.
                Defaults to None.
            custom_title (str, optional): Custom title for the plot. If None, a default title will
            be generated based on the column and split strategy. Defaults to None.
            dpi (int, optional): Resolution of the output image in dots per inch. Higher values
            result in better quality but larger file sizes. Defaults to 400.
            custom_high_low_names (Tuple[str, str], optional): Custom high and low variable names.
                Defaults to ("low", "high").


        Returns:
            dict: Dictionary containing the log-rank test p-value, plot filename, and test statistic.

        Note:
            This method splits the data into "high" and "low" groups based on the specified
            variable and strategy, then generates a Kaplan-Meier survival plot comparing
            the two groups. It also performs a log-rank test to compare the survival curves.
        """
        return generate_kaplan_meier_plot(
            self.df,
            column,
            split_strategy,
            fixed_value,
            percentage=percentage,
            output_path=output_path,
            custom_title=custom_title,
            dpi=dpi,
            custom_high_low_names=custom_high_low_names,
        )

    def multivariate_cox_regression(
        self, columns: List[str], penalizer: float = 0.1
    ) -> lifelines.CoxPHFitter:
        """
        Performs multivariate Cox proportional hazards regression.

        Args:
            columns (list): List of predictor column names.
            penalizer (float, optional): L2 penalizer value to apply to the regression. Defaults to 0.1.

        Returns:
            lifelines.CoxPHFitter: Fitted Cox proportional hazards model.

        Note:
            This method fits a Cox regression model with all specified variables simultaneously.
            It handles multicollinearity by iteratively removing variables with high VIF values.
            The standardize parameter from the class initialization determines whether
            variables are standardized before analysis.
        """
        return perform_multivariate_cox_regression(self.df, columns, penalizer, self.standardize)
