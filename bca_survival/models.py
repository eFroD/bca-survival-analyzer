"""
Survival Analysis Utilities Module.

This module provides functions for performing survival analysis, including Cox proportional hazards
regression models and Kaplan-Meier survival curves. It includes utilities for data preprocessing,
multicollinearity checking, and visualization of results.

Requires: pandas, numpy, scikit-learn, lifelines, matplotlib, statsmodels, seaborn
"""

import os
import warnings
from pathlib import Path
from typing import List, Optional, Union

import lifelines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.exceptions import ConvergenceError
from lifelines.statistics import logrank_test
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from tqdm.auto import tqdm

from bca_survival.utils import make_quantile_split


def standardize_columns(
    df: pd.DataFrame, columns: List[str], nan_threshold: float = 0.7
) -> pd.DataFrame:
    """
    Standardizes selected columns and handles missing values.

    Args:
        df (pd.DataFrame): The input dataframe.
        columns (list): List of column names to standardize.
        nan_threshold (float, optional): Threshold for NaN values. Columns with more NaNs
            than this threshold will be dropped. Defaults to 0.7.

    Returns:
        pd.DataFrame: DataFrame with standardized columns.

    Note:
        This function creates a copy of the dataframe and standardizes the specified
        columns using StandardScaler. Columns with too many NaN values are dropped.
    """
    for column in columns:
        nan_ratio = df[column].isna().mean()
        if nan_ratio > nan_threshold:
            print(f"Dropping column {column} due to {nan_ratio:.2%} NaNs")
            columns.remove(column)

    df = df.copy()
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df


def check_multicollinearity(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Checks multicollinearity between variables using a correlation matrix.

    Args:
        df (pd.DataFrame): The input dataframe.
        columns (list): List of column names to check for multicollinearity.

    Returns:
        pd.DataFrame: Correlation matrix of the specified columns.

    Note:
        This function also displays a heatmap of the correlation matrix.
    """
    corr_matrix = df[columns].corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()

    return corr_matrix


def perform_multivariate_cox_regression(
    df: pd.DataFrame,
    columns: List[str],
    penalizer: float = 0.1,
    standardize: bool = True,
    vif_threshold: float = 20,
) -> lifelines.CoxPHFitter:
    """
    Performs multivariate Cox proportional hazards regression.

    Args:
        df (pd.DataFrame): The input dataframe. Must contain 'days' and 'event' columns.
        columns (list): List of predictor column names.
        penalizer (float, optional): L2 penalizer value to apply to the regression. Defaults to 0.1.
        standardize (bool, optional): Whether to standardize the columns. Defaults to True.
        vif_threshold (float, optional): Threshold for Variance Inflation Factor (VIF).
            Variables with VIF above this threshold will be removed. Defaults to 20.

    Returns:
        lifelines.CoxPHFitter: Fitted Cox proportional hazards model.

    Note:
        This function handles multicollinearity by iteratively removing variables with
        high VIF values until all variables have VIF below the threshold.
    """
    if standardize:
        df = standardize_columns(df, columns)

    vif_data = calculate_vif(df, columns)
    print("Variance Inflation Factor (VIF) before removing variables:")
    print(vif_data)

    # Iteratively remove variables with high VIF
    while vif_data["VIF"].max() > vif_threshold:
        high_vif_vars = vif_data[vif_data["VIF"] > vif_threshold]["Variable"].tolist()
        for var in high_vif_vars:
            if var in columns:
                print(f"Removing variable with high VIF: {var}")
                columns.remove(var)
        vif_data = calculate_vif(df, columns)
        print("Updated VIF:")
        print(vif_data)
    # Fit the Cox Proportional Hazards model
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(df[["days", "event"] + columns].dropna(), duration_col="days", event_col="event")

    return cph


def perform_univariate_cox_regression(
    df: pd.DataFrame,
    columns: List[str],
    standardize: bool = False,
    penalizer: float = 0,
    verbose: bool = False,
    correction_values: Union[List[str], None] = None,
    nan_threshold: float = 0.7,
    significant_only: bool = True,
) -> pd.DataFrame:
    """
    Performs univariate Cox proportional hazards regression for each variable.

    Args:
        df (pd.DataFrame): The input dataframe. Must contain 'days' and 'event' columns.
        columns (list): List of predictor column names to test individually.
        standardize (bool, optional): Whether to standardize the columns. Defaults to False.
        penalizer (float, optional): L2 penalizer value to apply to the regression. Defaults to 0.
        verbose (bool, optional): Whether to print detailed progress information. Defaults to False.
        correction_values (list, optional): List of column names to include as correction terms in each
            univariate model. Often you'll use this to correct for age or gender effects. Defaults to None.
        nan_threshold (float, optional): Threshold for NaN values if standardizing. Defaults to 0.7.
        significant_only (bool, optional): Whether to only include significant values. Defaults to True.

    Returns:
        pd.DataFrame: DataFrame containing significant variables and their statistics.

    Note:
        This function tests each variable individually in a Cox regression model,
        and returns only statistically significant variables (p < 0.05).
    """
    if standardize:
        df = standardize_columns(df, columns, nan_threshold=nan_threshold)
    significant_variables = []
    cph = CoxPHFitter(penalizer=penalizer)

    if correction_values is None:
        correction_values = []

    for column in tqdm(columns, desc="Analyzing Columns"):
        df_temp = df[[column, "days", "event"] + correction_values]
        len_before = len(df_temp)
        df_temp = df_temp.dropna()
        len_after = len(df_temp)
        if verbose:
            print(f"Analyzing column: {column}")
            if len_after < len_before:
                print(f"Removed {len_before - len_after} nan rows.")

        if df_temp[column].dtype.name != "category":
            if df_temp[column].mean() == df_temp[column].std() == 0:
                if verbose:
                    print("Zero mean and zero variance. Skipping.")
                continue
        if len(df_temp) < 10:
            if verbose:
                print("Too few observations. Skipping.")
            continue
        if "WL" in column and len(df_temp[column].value_counts()) < 5:
            if verbose:
                print("Too few unique values in BCA. Skipping")
            continue
        warning = ""
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                cph.fit(df_temp, duration_col="days", event_col="event")

                if w:
                    warning = "yes"
            summary = cph.summary
            c_index = cph.concordance_index_
            log_likelihood = cph.log_likelihood_
            aic = cph.AIC_partial_
            # bic = cph.BIC_
            if summary["p"].values[0] < 0.05 or not significant_only:
                if verbose:
                    print(summary)
                significant_variables.append(
                    {
                        "Variable": column,
                        "HR": summary["exp(coef)"].values[0],
                        "p-value": summary["p"].values[0],
                        "95% lower-bound": summary["exp(coef) lower 95%"].iloc[0],
                        "95% upper-bound": summary["exp(coef) upper 95%"].iloc[0],
                        "n": len(df_temp),
                        "c_index": c_index,
                        "log_likelihood": log_likelihood,
                        "aic": aic,
                        # "bic": bic,
                        "convergence warning": warning,
                        "correction_terms": correction_values,
                        #  'summary': summary
                    }
                )
        except ConvergenceError:
            warnings.warn("Convergence error encountered for column {}, skipping.".format(column))
    significant_df = pd.DataFrame(significant_variables)
    return significant_df


def generate_kaplan_meier_plot(
    df: pd.DataFrame,
    column: str,
    split_strategy: str = "median",
    fixed_value: Optional[float] = None,
    percentage: Optional[float] = None,
    output_path: Optional[Union[os.PathLike[str], str]] = None,
    dpi: int = 600,
    custom_title: Optional[str] = None,
    display_plot: bool = False,
) -> dict:
    """
    Generates a Kaplan-Meier survival plot for a specified variable.

    Args:
        df (pd.DataFrame): The input dataframe. Must contain 'days' and 'event' columns.
        column (str): Column name to use for grouping.
        split_strategy (str, optional): Strategy for splitting data into high/low groups.
            Options: 'mean', 'median', 'percentage', 'fixed', 'quantile'. Defaults to 'median'.
        fixed_value (float, optional): Fixed threshold value when split_strategy is 'fixed'.
            You can use this when you have found cutoff values from literature.
            Defaults to None.
        percentage (float, optional): Percentile threshold when split_strategy is 'percentage'.
            Defaults to None.
        output_path (str, optional): Directory path to save the plot. If None, saves in current
            directory. Defaults to None.
        dpi (int, optional): Resolution of the output image in dots per inch. Higher values
            result in better quality but larger file sizes. Defaults to 600.
        custom_title (str, optional): Custom title for the plot. If None, a default title will
            be generated based on the column and split strategy. Defaults to None.
        display_plot (bool, optional): Whether to display the plot in the notebook. If False,
            the plot is only saved to file without rendering. Defaults to False.

    Returns:
        dict: Dictionary containing the log-rank test p-value, plot filename, and test statistic.

    Raises:
        ValueError: If an invalid split_strategy is provided or if required parameters for a
            particular strategy are missing.

    Note:
        This function splits the data into "high" and "low" groups based on the specified
        variable and strategy, then generates a Kaplan-Meier survival plot comparing
        the two groups. It also performs a log-rank test to compare the survival curves.
    """
    import matplotlib.pyplot as plt

    # For optimization in notebooks when display_plot is False
    if not display_plot:
        # Use plt.ioff() to turn off interactive mode
        plt.ioff()
    else:
        plt.ion()

    if split_strategy == "mean":
        threshold = df[column].mean()
    elif split_strategy == "median":
        threshold = df[column].median()
    elif split_strategy == "percentage":
        threshold = df[column].quantile(percentage)
    elif split_strategy == "fixed" and fixed_value is not None:
        threshold = fixed_value
    elif split_strategy == "quantile":
        threshold = "quantile"
    else:
        raise ValueError(
            "Invalid split_strategy. Use 'mean', 'median', 'percentage', 'fixed', or 'quantile'. "
            "For 'fixed', provide fixed_value. For 'percentage', provide percentage."
        )
    df_tmp = df.copy().dropna(subset=column)
    if threshold == "quantile":
        df_tmp = make_quantile_split(df_tmp, column)
    else:
        df_tmp["group"] = np.where(df_tmp[column] > threshold, "high", "low")

    # Create figure
    fig, ax = plt.subplots()

    kmf = KaplanMeierFitter()
    results_high = df_tmp[df_tmp["group"] == "high"]
    results_low = df_tmp[df_tmp["group"] == "low"]

    kmf.fit(durations=results_high["days"], event_observed=results_high["event"], label="high")
    kmf.plot_survival_function(ax=ax)

    kmf.fit(durations=results_low["days"], event_observed=results_low["event"], label="low")
    kmf.plot_survival_function(ax=ax)

    # Use custom title if provided, otherwise use default
    if custom_title:
        ax.set_title(custom_title)
    else:
        ax.set_title(f"Survival function by {column} ({split_strategy} split)")

    ax.set_xlabel("Days")
    ax.set_ylabel("Survival probability")

    logrank_results = logrank_test(
        results_high["days"], results_low["days"], results_high["event"], results_low["event"]
    )
    p_value = logrank_results.p_value
    fig.text(0.15, 0.2, f"p-value: {p_value:.4f}", fontsize=12, ha="left")

    plot_filename = f"km_plot_{column}_{split_strategy}.png"
    plot_filename = (
        plot_filename.replace(" ", "_").replace("\n", "").replace("/", "").replace(":", "_")
    )

    if output_path:
        Path(output_path).mkdir(exist_ok=True, parents=True)
        fig.savefig(str(Path(output_path, plot_filename)), dpi=dpi)
    else:
        fig.savefig(str(plot_filename), dpi=dpi)

    plt.close(fig)

    # Restore interactive mode if needed
    if not display_plot:
        plt.ion()

    return {
        "p-value": p_value,
        "plot_filename": plot_filename,
        "metrics": logrank_results.test_statistic,
    }


def calculate_vif(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Calculates the Variance Inflation Factor (VIF) for each variable.

    Args:
        df (pd.DataFrame): The input dataframe.
        columns (list): List of column names to calculate VIF for.

    Returns:
        pd.DataFrame: DataFrame containing variables and their corresponding VIF values.

    Note:
        VIF is a measure of multicollinearity. Higher values indicate stronger
        correlation with other variables. VIF > 10 is often considered problematic.
    """
    # Add a constant for VIF calculation
    df_with_const = df.copy()
    df_with_const["constant"] = 1

    # Calculate VIF for each variable
    vif_data = pd.DataFrame()
    vif_data["Variable"] = columns
    vif_data["VIF"] = [
        variance_inflation_factor(df_with_const[columns + ["constant"]].dropna().values, i)
        for i in range(len(columns))
    ]

    return vif_data
