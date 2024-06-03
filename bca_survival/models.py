import pandas as pd
import numpy as np
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def standardize_columns(df, columns):
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df


def perform_univariate_cox_regression(df, columns, standardize=False):
    if standardize:
        df = standardize_columns(df, columns)
    significant_variables = []
    cph = CoxPHFitter()
    for column in columns:
        df_temp = df[[column, 'days', 'status']].dropna()
        cph.fit(df_temp, duration_col='days', event_col='status')
        summary = cph.summary
        if summary['p'].values[0] < 0.05:  # Significance level of 0.05
            significant_variables.append({
                'Variable': column,
                'HR': summary['exp(coef)'].values[0],
                'p-value': summary['p'].values[0]
            })
    significant_df = pd.DataFrame(significant_variables)
    return significant_df


def generate_kaplan_meier_plot(df, column, split_strategy='median', fixed_value=None):
    if split_strategy == 'mean':
        threshold = df[column].mean()
    elif split_strategy == 'median':
        threshold = df[column].median()
    elif split_strategy == 'fixed' and fixed_value is not None:
        threshold = fixed_value
    else:
        raise ValueError("Invalid split_strategy. Use 'mean', 'median', or 'fixed'. For 'fixed', provide fixed_value.")
    df['group'] = np.where(df[column] > threshold, 'high', 'low')
    kmf = KaplanMeierFitter()
    results_high = df[df['group'] == 'high']
    results_low = df[df['group'] == 'low']
    kmf.fit(durations=results_high['days'], event_observed=results_high['status'], label='high')
    ax = kmf.plot_survival_function()
    kmf.fit(durations=results_low['days'], event_observed=results_low['status'], label='low')
    kmf.plot_survival_function(ax=ax)
    plt.title(f'Survival function by {column} ({split_strategy} split)')
    plt.xlabel('Days')
    plt.ylabel('Survival probability')
    logrank_results = logrank_test(results_high['days'], results_low['days'], results_high['status'], results_low['status'])
    p_value = logrank_results.p_value
    plt.figtext(0.15, 0.2, f'p-value: {p_value:.4f}', fontsize=12, ha='left')
    plot_filename = f'km_plot_{column}_{split_strategy}.png'
    plt.savefig(plot_filename)
    plt.show()
    return {
        'p-value': p_value,
        'plot_filename': plot_filename,
        'metrics': logrank_results.test_statistic
    }
