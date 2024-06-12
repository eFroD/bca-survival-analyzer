import warnings

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.exceptions import ConvergenceError
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def standardize_columns(df, columns):
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df


def perform_univariate_cox_regression(df, columns, standardize=False, penalizer=0.1, verbose=False):
    if standardize:
        df = standardize_columns(df, columns)
    significant_variables = []
    cph = CoxPHFitter(penalizer=penalizer)
    for column in tqdm(columns, desc="Analyzing Columns"):
        if verbose:
            print(f"Analyzing column: {column}")
        df_temp = df[[column, 'days', 'event']].dropna()
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
                cph.fit(df_temp, duration_col='days', event_col='event')

                if w:
                    warning = "yes"
            summary = cph.summary
            if summary['p'].values[0] < 0.05:
                significant_variables.append({
                    'Variable': column,
                    'HR': summary['exp(coef)'].values[0],
                    'p-value': summary['p'].values[0],
                    'convergence warning': warning
                })
        except ConvergenceError:
            warnings.warn("Convergence error encountered for column {}, skipping.".format(column))
    significant_df = pd.DataFrame(significant_variables)
    return significant_df


def generate_kaplan_meier_plot(df, column, split_strategy='median', fixed_value=None, output_path=None):
    if split_strategy == 'mean':
        threshold = df[column].mean()
    elif split_strategy == 'median':
        threshold = df[column].median()
    elif split_strategy == 'fixed' and fixed_value is not None:
        threshold = fixed_value
    else:
        raise ValueError("Invalid split_strategy. Use 'mean', 'median', or 'fixed'. For 'fixed', provide fixed_value.")
    df_tmp = df.copy().dropna(subset=column)
    df_tmp['group'] = np.where(df_tmp[column] > threshold, 'high', 'low')
    kmf = KaplanMeierFitter()
    results_high = df_tmp[df_tmp['group'] == 'high']
    results_low = df_tmp[df_tmp['group'] == 'low']
    kmf.fit(durations=results_high['days'], event_observed=results_high['event'], label='high')
    ax = kmf.plot_survival_function()
    kmf.fit(durations=results_low['days'], event_observed=results_low['event'], label='low')
    kmf.plot_survival_function(ax=ax)
    plt.title(f'Survival function by {column} ({split_strategy} split)')
    plt.xlabel('Days')
    plt.ylabel('Survival probability')
    logrank_results = logrank_test(results_high['days'], results_low['days'], results_high['event'], results_low['event'])
    p_value = logrank_results.p_value
    plt.figtext(0.15, 0.2, f'p-value: {p_value:.4f}', fontsize=12, ha='left')
    plot_filename = f'km_plot_{column}_{split_strategy}.png'
    plot_filename = (plot_filename.replace(' ', '_').replace('\n', '')
                     .replace('/', '').replace(':', '_'))
    if output_path:
        plt.savefig(str(Path(output_path, plot_filename)))
    else:
        plt.savefig(str(plot_filename))
    plt.show()
    return {
        'p-value': p_value,
        'plot_filename': plot_filename,
        'metrics': logrank_results.test_statistic
    }
