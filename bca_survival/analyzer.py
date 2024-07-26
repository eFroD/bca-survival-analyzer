import pandas as pd
import numpy as np
from .preprocessing import calculate_days, check_and_remove_negative_days
from .models import perform_univariate_cox_regression, generate_kaplan_meier_plot, perform_multivariate_cox_regression


class BCASurvivalAnalyzer:
    def __init__(self, df_main, df_measurements, main_id_col, measurement_id_col,
                 start_date_col, event_date_col=None, event_col=None, standardize=False):
        # Rename ID columns to unify them across both dataframes
        df_main = df_main.rename(columns={main_id_col: 'PID'})
        df_measurements = df_measurements.rename(columns={measurement_id_col: 'PID'})
        df_main.replace('nd', np.nan, inplace=True)
        df_measurements.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Merge the main and measurements dataframes on the unified ID column
        self.df = pd.merge(df_main, df_measurements, on='PID', how='left')
        # self.df = self.df.dropna(subset='event_date')
        # self.df = self.df.dropna(subset='event')
        self.start_date_col = start_date_col
        self.event_date_col = event_date_col
        self.event_col = event_col
        self.standardize = standardize
        self.preprocess_data()

    def preprocess_data(self):
        self.df = calculate_days(self.df, self.start_date_col, self.event_date_col, self.event_col)
        self.df = check_and_remove_negative_days(self.df)
        return self.df

    def univariate_cox_regression(self, columns, verbose=False, penalizer=0.1, correction_values=None):
        if correction_values is None:
            correction_values = []
        return perform_univariate_cox_regression(self.df, columns, self.standardize, verbose=verbose, penalizer=penalizer, correction_values=correction_values)

    def kaplan_meier_plot(self, column, split_strategy='median', fixed_value=None, output_path=None, percentage=None):
        return generate_kaplan_meier_plot(self.df, column, split_strategy, fixed_value, percentage=percentage, output_path=output_path)

    def multivariate_cox_regression(self, columns, penalizer=0.1):
        return perform_multivariate_cox_regression(self.df, columns, penalizer, self.standardize)
