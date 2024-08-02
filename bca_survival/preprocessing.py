import pandas as pd


def calculate_days(df, start_date_col, event_date_col, event_col):
    if event_date_col:
        df['days'] = (pd.to_datetime(df[event_date_col], format='%d.%m.%Y') - pd.to_datetime(df[start_date_col], format='%d.%m.%Y')).dt.days
    df['event'] = df[event_col].astype(int)

    return df


def check_and_remove_negative_days(df):
    negative_values_count = (df['days'] < 0).sum()
    df_negative = None
    if negative_values_count > 0:
        print(
            f"Warning: {negative_values_count} rows will be dropped because they contain values below 0 in the 'days' column.")
        df_negative = df[df['days'] < 0]
        df = df[df['days'] >= 0]

    return df, df_negative


def create_event_date_column(df, date_death, date_disease_death, date_followup):
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
    # Define the parts and metrics
    body_parts = ['ventral_cavity', 'abdominal_cavity', 'thoracic_cavity', 'mediastinum', 'pericardium', 'l5', 'l4',
                  'l3', 'l2', 'l1', 't12', 't11', 't10', 't9', 't8', 't7', 't6', 't5', 't4', 't3', 't2', 't1']
    metrics = [
        "mean_ml",
        "std_ml",
        "min_ml",
        "q1_ml",
        "q2_ml",
        "q3_ml",
        "max_ml",
        "sum_ml",
        "mean_hu"
    ]

    ratios = [('imat', 'tat'), ('vat', 'tat'), ('eat', 'tat'), ('sat', 'tat'), ('pat', 'tat'), ('muscle', 'bone')]

    # Iterate through each body part and metric combination
    for body_part in body_parts:
        for metric in metrics:
            for (numerator, denominator) in ratios:
                num_col = f'{body_part}::WL::{numerator}::{metric}'
                den_col = f'{body_part}::WL::{denominator}::{metric}'
                new_col = f'{body_part}::WL::{numerator}/{denominator}::{metric}'

                if num_col in df.columns and den_col in df.columns:
                    df[new_col] = df[num_col] / df[den_col]

    return df
