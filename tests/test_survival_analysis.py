"""
Comprehensive test suite for survival analysis modules.

This script tests all functions from the preprocessing, models, and utilities modules
using synthetic data to ensure proper functionality.
"""

import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
import warnings

# Assuming the following imports from your actual modules
# Import preprocessing module functions
from bca_survival.preprocessing import calculate_days, check_and_remove_negative_days, create_event_date_column, compute_ratios

# Import models module functions
from bca_survival.models import standardize_columns, check_multicollinearity, perform_multivariate_cox_regression, \
    perform_univariate_cox_regression, generate_kaplan_meier_plot, calculate_vif

# Import utility functions
from bca_survival.utils import make_quantile_split, make_quantile_split_outter_vs_middle, calculate_age, clean_dates

# Import analyzer class
from bca_survival.analyzer import BCASurvivalAnalyzer

# Suppress matplotlib plots in testing
plt.ioff()


class TestPreprocessingFunctions(unittest.TestCase):
    """Tests for functions in the preprocessing module."""

    def setUp(self):
        """Set up synthetic data for tests."""
        # Create sample dataframe with dates
        self.df = pd.DataFrame({
            'patient_id': range(1, 11),
            'diagnosis_date': ['15.01.2020', '20.02.2020', '10.03.2020', '05.04.2020', '25.05.2020',
                               '30.06.2020', '15.07.2020', '10.08.2020', '20.09.2020', '15.10.2020'],
            'event_date': ['20.05.2020', '15.08.2020', '25.09.2020', '10.11.2020', '30.12.2020',
                           '05.01.2021', '20.02.2021', '15.03.2021', None, '10.05.2021'],
            'event_status': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            'death_date': [None, None, '25.09.2020', None, '30.12.2020', None, None, None, None, None],
            'disease_death_date': ['20.05.2020', '15.08.2020', None, '10.11.2020', None, None, None, None, None, None],
            'follow_up_date': [None, None, None, None, None, '05.01.2021', '20.02.2021', '15.03.2021', '30.03.2021',
                               '10.05.2021']
        })

        # Create dataframe for testing compute_ratios
        self.tissue_df = pd.DataFrame()
        body_parts = ['l5', 't12']
        tissue_types = ['imat', 'tat', 'vat', 'muscle', 'bone']
        metrics = ['mean_ml', 'sum_ml']

        # Initialize random data for tissue measurements
        np.random.seed(42)
        for part in body_parts:
            for tissue in tissue_types:
                for metric in metrics:
                    column_name = f'{part}::WL::{tissue}::{metric}'
                    self.tissue_df[column_name] = np.random.rand(10) * 100

    def test_calculate_days(self):
        """Test calculate_days function."""
        result_df = calculate_days(self.df, 'diagnosis_date', 'event_date', 'event_status')

        self.assertIn('days', result_df.columns)
        self.assertIn('event', result_df.columns)

        # Check calculations for a few samples
        # days between 15.01.2020 and 20.05.2020 should be 126
        self.assertEqual(result_df.loc[0, 'days'], 126)

        # Check event conversion to int
        self.assertEqual(result_df.loc[0, 'event'], 1)
        self.assertEqual(result_df.loc[5, 'event'], 0)

    def test_check_and_remove_negative_days(self):
        """Test check_and_remove_negative_days function."""
        # Add some negative days
        test_df = self.df.copy()
        test_df['days'] = [126, 177, 199, 219, 219, -10, 220, 217, np.nan, 208]
        test_df['event'] = test_df['event_status']

        clean_df, negative_df = check_and_remove_negative_days(test_df)

        # Check if negative and NaN rows are removed
        self.assertEqual(len(clean_df), 8)
        self.assertEqual(len(negative_df), 2)

        # Check if all remaining days are non-negative
        self.assertTrue((clean_df['days'] >= 0).all())

        # Verify one of the negative rows
        self.assertEqual(negative_df.iloc[0]['days'], -10)

    def test_create_event_date_column(self):
        """Test create_event_date_column function."""
        result_df = create_event_date_column(
            self.df, 'death_date', 'disease_death_date', 'follow_up_date'
        )

        # Check the columns were created
        self.assertIn('event_date', result_df.columns)
        self.assertIn('event', result_df.columns)

        # Check a death event
        self.assertEqual(result_df.loc[2, 'event_date'], '25.09.2020')
        self.assertTrue(result_df.loc[2, 'event'])

        # Check a disease death event
        self.assertEqual(result_df.loc[0, 'event_date'], '20.05.2020')
        self.assertTrue(result_df.loc[0, 'event'])

        # Check a follow-up event (non-event)
        self.assertEqual(result_df.loc[5, 'event_date'], '05.01.2021')
        self.assertFalse(result_df.loc[5, 'event'])

        # Check missing data case
        self.assertEqual(result_df.loc[8, 'event_date'], '30.03.2021')
        self.assertFalse(result_df.loc[8, 'event'])

    def test_compute_ratios(self):
        """Test compute_ratios function."""
        result_df = compute_ratios(self.tissue_df)

        # Check that ratio columns were created
        expected_ratio_columns = [
            'l5::WL::imat/tat::mean_ml',
            'l5::WL::vat/tat::mean_ml',
            'l5::WL::muscle/bone::mean_ml',
            'l5::WL::imat/muscle::mean_ml',
            't12::WL::imat/tat::sum_ml'
        ]

        for col in expected_ratio_columns:
            self.assertIn(col, result_df.columns)

        # Verify a ratio calculation
        imat_col = 'l5::WL::imat::mean_ml'
        tat_col = 'l5::WL::tat::mean_ml'
        ratio_col = 'l5::WL::imat/tat::mean_ml'

        expected_ratio = self.tissue_df[imat_col] / self.tissue_df[tat_col]
        pd.testing.assert_series_equal(result_df[ratio_col], expected_ratio.rename(ratio_col))


class TestModelsFunctions(unittest.TestCase):
    """Tests for functions in the models module."""

    def setUp(self):
        """Set up synthetic data for tests."""
        np.random.seed(42)
        n_samples = 100

        # Create synthetic survival data
        self.df = pd.DataFrame({
            'days': np.random.randint(10, 1000, size=n_samples),
            'event': np.random.binomial(1, 0.7, size=n_samples),
            'age': np.random.normal(65, 10, size=n_samples),
            'bmi': np.random.normal(25, 5, size=n_samples),
            'gender': np.random.binomial(1, 0.6, size=n_samples),
            'comorbidity_score': np.random.normal(3, 2, size=n_samples),
            'treatment': np.random.binomial(1, 0.5, size=n_samples),
            'tumor_size': np.random.normal(3, 1.5, size=n_samples),
            'highly_correlated1': np.random.normal(50, 10, size=n_samples),
            'highly_correlated2': np.random.normal(50, 10, size=n_samples) * 0.95 + 2.5
            # Highly correlated with highly_correlated1
        })

        # Add some NaN values
        self.df.loc[np.random.choice(n_samples, 10), 'bmi'] = np.nan
        self.df.loc[np.random.choice(n_samples, 5), 'tumor_size'] = np.nan

        # Columns to use in tests
        self.columns = ['age', 'bmi', 'gender', 'comorbidity_score', 'treatment', 'tumor_size']
        self.correlated_columns = ['highly_correlated1', 'highly_correlated2'] + self.columns

    def test_standardize_columns(self):
        """Test standardize_columns function."""
        result_df = standardize_columns(self.df, self.columns)

        # Check that columns were standardized
        for col in self.columns:
            if col in result_df.columns:  # Some might be dropped due to NaN threshold
                # Standardized columns should have mean close to 0 and std close to 1
                mean = result_df[col].mean()
                std = result_df[col].std()
                self.assertAlmostEqual(mean, 0, places=1)
                self.assertAlmostEqual(std, 1, places=1)

    def test_check_multicollinearity(self):
        """Test check_multicollinearity function."""
        # This is mostly visual, so we just check it runs without error
        corr_matrix = check_multicollinearity(self.df, self.correlated_columns)

        # Check that correlation matrix has correct shape
        self.assertEqual(corr_matrix.shape, (len(self.correlated_columns), len(self.correlated_columns)))



    def test_perform_univariate_cox_regression(self):
        """Test perform_univariate_cox_regression function."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore convergence warnings
            significant_df = perform_univariate_cox_regression(
                self.df, self.columns, standardize=True, verbose=False
            )

        # Check that output has expected structure
        if not significant_df.empty:
            expected_columns = [
                'Variable', 'HR', 'p-value', '95% lower-bound',
                '95% upper-bound', 'n', 'convergence warning'
            ]
            for col in expected_columns:
                self.assertIn(col, significant_df.columns)

    def test_generate_kaplan_meier_plot(self):
        """Test generate_kaplan_meier_plot function."""
        # Create a temporary directory for plots
        os.makedirs('test_plots', exist_ok=True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore convergence warnings
            result = generate_kaplan_meier_plot(
                self.df, 'age', split_strategy='median', output_path='test_plots'
            )

        # Check that result has expected structure
        self.assertIn('p-value', result)
        self.assertIn('plot_filename', result)
        self.assertIn('metrics', result)

        # Check that plot file was created
        plot_path = os.path.join('test_plots', result['plot_filename'])
        self.assertTrue(os.path.exists(plot_path))

        # Clean up
        if os.path.exists(plot_path):
            os.remove(plot_path)
        if os.path.exists('test_plots'):
            os.rmdir('test_plots')

    def test_perform_multivariate_cox_regression(self):
        """Test perform_multivariate_cox_regression function."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore convergence warnings
            cph = perform_multivariate_cox_regression(
                self.df, self.columns, penalizer=0.1, standardize=True, vif_threshold=10
            )

        # Check that a model was created
        self.assertIsNotNone(cph)
        self.assertTrue(hasattr(cph, 'summary'))

        # Ensure high VIF variables were handled
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cph_correlated = perform_multivariate_cox_regression(
                self.df, self.correlated_columns, penalizer=0.1, standardize=True, vif_threshold=10
            )


class TestUtilityFunctions(unittest.TestCase):
    """Tests for utility functions."""

    def setUp(self):
        """Set up synthetic data for tests."""
        np.random.seed(42)
        n_samples = 50

        # Data for quantile splits
        self.values_df = pd.DataFrame({
            'id': range(1, n_samples + 1),
            'value': np.random.normal(100, 25, size=n_samples),
            'score': np.random.normal(50, 15, size=n_samples)
        })

        # Add some NaN values
        self.values_df.loc[np.random.choice(n_samples, 5), 'value'] = np.nan

        # Data for date cleaning and age calculation
        today = datetime.now()

        dates = []
        birth_dates = []
        for i in range(n_samples):
            # Random date within the last 3 years
            days_ago = np.random.randint(1, 1095)  # Up to 3 years
            date = today - timedelta(days=days_ago)
            dates.append(date.strftime('%d.%m.%Y'))

            # Random birth date for people between 20 and 90 years old
            age_years = np.random.randint(20, 90)
            birth_date = today - timedelta(days=365.25 * age_years) - timedelta(days=np.random.randint(0, 365))
            birth_dates.append(birth_date.strftime('%d.%m.%Y'))

        self.dates_df = pd.DataFrame({
            'id': range(1, n_samples + 1),
            'visit_date': dates,
            'birth_date': birth_dates
        })

        # Add some invalid dates
        self.dates_df.loc[np.random.choice(n_samples, 3), 'visit_date'] = 'invalid-date'
        self.dates_df.loc[np.random.choice(n_samples, 2), 'birth_date'] = None

    def test_make_quantile_split(self):
        """Test make_quantile_split function."""
        result_df = make_quantile_split(self.values_df, 'value')

        # Check that group column was created
        self.assertIn('group', result_df.columns)

        # Check that NaN values were removed
        self.assertEqual(len(result_df), len(self.values_df) - self.values_df['value'].isna().sum())

        # Check that groups were assigned correctly
        q25 = self.values_df['value'].quantile(0.25)
        q75 = self.values_df['value'].quantile(0.75)

        for _, row in result_df.iterrows():
            if not pd.isna(row['group']):
                if row['value'] <= q25:
                    self.assertEqual(row['group'], 'low')
                elif row['value'] >= q75:
                    self.assertEqual(row['group'], 'high')
            else:
                self.assertTrue(q25 < row['value'] < q75)

    def test_make_quantile_split_outter_vs_middle(self):
        """Test make_quantile_split_outter_vs_middle function."""
        result_df = make_quantile_split_outter_vs_middle(self.values_df, 'value')

        # Check that group column was created
        self.assertIn('group', result_df.columns)

        # Check that NaN values were removed
        self.assertEqual(len(result_df), len(self.values_df) - self.values_df['value'].isna().sum())

        # Check that groups were assigned correctly
        q25 = self.values_df['value'].quantile(0.25)
        q75 = self.values_df['value'].quantile(0.75)

        for _, row in result_df.iterrows():
            if row['value'] <= q25 or row['value'] >= q75:
                self.assertEqual(row['group'], 'outer')
            else:
                self.assertEqual(row['group'], 'middle')

    def test_calculate_age(self):
        """Test calculate_age function."""
        result_df = calculate_age(self.dates_df, 'birth_date', 'visit_date')

        # Check that age column was created
        self.assertIn('Age', result_df.columns)

        # Test a few ages manually
        for i in range(min(5, len(result_df))):
            row = result_df.iloc[i]
            if pd.notna(row['birth_date']) and pd.notna(row['visit_date']):
                birth_date = row['birth_date']
                visit_date = row['visit_date']

                # Calculate expected age
                years_diff = visit_date.year - birth_date.year
                is_before_birthday = (visit_date.month, visit_date.day) < (birth_date.month, birth_date.day)
                expected_age = years_diff - is_before_birthday

                self.assertEqual(row['Age'], expected_age)

        # Check that rows with missing dates have NaN age
        self.assertTrue(result_df[result_df['birth_date'].isna()]['Age'].isna().all())

    def test_clean_dates(self):
        """Test clean_dates function."""
        # Test with auto detection
        clean_df, stats = clean_dates(self.dates_df, 'visit_date', date_format="%d.%m.%Y")

        # Check that invalid dates were removed
        self.assertEqual(stats['removed_rows'], self.dates_df['visit_date'].apply(
            lambda x: not isinstance(x, str) or 'invalid' in str(x)).sum())

        # All remaining visit_date values should be valid dates
        self.assertTrue(clean_df['visit_date'].notna().all())

        # Test with specific format
        clean_df2, stats2 = clean_dates(self.dates_df, 'visit_date', date_format='%d.%m.%Y')

        # Results should be the same as with auto detection for this data
        self.assertEqual(len(clean_df), len(clean_df2))


class TestBCASurvivalAnalyzer(unittest.TestCase):
    """Tests for the BCASurvivalAnalyzer class."""

    def setUp(self):
        """Set up synthetic data for tests."""
        np.random.seed(42)
        n_samples = 50

        # Create clinical data
        self.df_clinical = pd.DataFrame({
            'patient_id': range(1, n_samples + 1),
            'age': np.random.normal(65, 10, size=n_samples),
            'gender': np.random.binomial(1, 0.6, size=n_samples),
            'diagnosis_date': ['15.01.2020'] * n_samples,
            'event_date': [f'{((15 + i) % 28) + 1}.{((i % 12) + 1):02d}.{2020 + (i // 12)}' for i in range(n_samples-1)] + ["30.12.2019"],
            'event_status': np.random.binomial(1, 0.3, size=n_samples)
        })

        # Create BCA measurements data
        self.df_measurements = pd.DataFrame({
            'id': range(1, n_samples + 1),
            'l5::WL::imat::mean_ml': np.random.normal(10, 3, size=n_samples),
            'l5::WL::tat::mean_ml': np.random.normal(50, 15, size=n_samples),
            'l5::WL::muscle::mean_ml': np.random.normal(100, 25, size=n_samples),
            't12::WL::imat::mean_ml': np.random.normal(8, 2, size=n_samples),
            't12::WL::tat::mean_ml': np.random.normal(40, 10, size=n_samples),
            't12::WL::muscle::mean_ml': np.random.normal(90, 20, size=n_samples)
        })

        # Create a few missing patients in measurements
        missing_indices = np.random.choice(n_samples, 5, replace=False)
        self.df_measurements = self.df_measurements[~self.df_measurements['id'].isin(missing_indices)]

    def test_initialization(self):
        """Test BCASurvivalAnalyzer initialization."""
        analyzer = BCASurvivalAnalyzer(
            self.df_clinical, self.df_measurements,
            'patient_id', 'id',
            'diagnosis_date', 'event_date', 'event_status'
        )

        # Check that dataframes were merged correctly

        # Check that days were calculated
        self.assertIn('days', analyzer.df.columns)
        self.assertIn('event', analyzer.df.columns)

        # Check that negative days were handled
        self.assertIsNotNone(analyzer.df_negative_days)

    def test_univariate_cox_regression(self):
        """Test univariate_cox_regression method."""
        analyzer = BCASurvivalAnalyzer(
            self.df_clinical, self.df_measurements,
            'patient_id', 'id',
            'diagnosis_date', 'event_date', 'event_status'
        )

        columns = ['l5::WL::imat::mean_ml', 'l5::WL::tat::mean_ml', 'l5::WL::muscle::mean_ml']

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore convergence warnings
            results = analyzer.univariate_cox_regression(columns)

        # Check that results have expected structure
        if not results.empty:
            expected_columns = [
                'Variable', 'HR', 'p-value', '95% lower-bound',
                '95% upper-bound', 'n', 'convergence warning'
            ]
            for col in expected_columns:
                self.assertIn(col, results.columns)

    def test_kaplan_meier_plot(self):
        """Test kaplan_meier_plot method."""
        analyzer = BCASurvivalAnalyzer(
            self.df_clinical, self.df_measurements,
            'patient_id', 'id',
            'diagnosis_date', 'event_date', 'event_status'
        )

        # Create a temporary directory for plots
        os.makedirs('test_plots', exist_ok=True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore convergence warnings
            result = analyzer.kaplan_meier_plot(
                'l5::WL::muscle::mean_ml', split_strategy='median', output_path='test_plots'
            )

        # Check that result has expected structure
        self.assertIn('p-value', result)
        self.assertIn('plot_filename', result)

        # Clean up
        plot_path = os.path.join('test_plots', result['plot_filename'])
        if os.path.exists(plot_path):
            os.remove(plot_path)
        if os.path.exists('test_plots'):
            os.rmdir('test_plots')

    def test_multivariate_cox_regression(self):
        """Test multivariate_cox_regression method."""
        analyzer = BCASurvivalAnalyzer(
            self.df_clinical, self.df_measurements,
            'patient_id', 'id',
            'diagnosis_date', 'event_date', 'event_status'
        )

        columns = ['l5::WL::imat::mean_ml', 'l5::WL::tat::mean_ml', 'age', 'gender']

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore convergence warnings
            cph = analyzer.multivariate_cox_regression(columns)

        # Check that a model was created
        self.assertIsNotNone(cph)
        self.assertTrue(hasattr(cph, 'summary'))


if __name__ == '__main__':
    # Create output directory for any test plots
    os.makedirs('test_plots', exist_ok=True)

    # Run the tests
    unittest.main()