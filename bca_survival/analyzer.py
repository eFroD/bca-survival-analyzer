import pandas as pd
from .preprocessing import preprocess_data
from .models import perform_univariate_cox_regression, generate_kaplan_meier_plot


class SurvivalAnalysisFramework:
    def __init__(self, df_main, df_measurements, main_id_col, measurement_id_col,
                 start_date_col, event_date_col=None, event_col=None, standardize=False):
        # Rename ID columns to unify them across both dataframes
        df_main = df_main.rename(columns={main_id_col: 'ID'})
        df_measurements = df_measurements.rename(columns={measurement_id_col: 'ID'})
        # Merge the main and measurements dataframes on the unified ID column
        self.df = pd.merge(df_main, df_measurements, on='ID', how='left')
        self.start_date_col = start_date_col
        self.event_date_col = event_date_col
        self.event_col = event_col
        self.standardize = standardize

    def preprocess_data(self):
        self.df = preprocess_data(self.df, self.start_date_col, self.event_date_col, self.event_col)
        return self.df

    def univariate_cox_regression(self, columns):
        return perform_univariate_cox_regression(self.df, columns, self.standardize)

    def kaplan_meier_plot(self, column, split_strategy='median', fixed_value=None):
        return generate_kaplan_meier_plot(self.df, column, split_strategy, fixed_value)
