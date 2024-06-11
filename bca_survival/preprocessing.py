import pandas as pd


def calculate_days(df, start_date_col, event_date_col, event_col):
    if event_date_col:
        df['days'] = (pd.to_datetime(df[event_date_col], format='%d.%m.%Y') - pd.to_datetime(df[start_date_col], format='%d.%m.%Y')).dt.days
    df['event'] = df[event_col].astype(int)

    return df


def check_and_remove_negative_days(df):
    negative_values_count = (df['days'] < 0).sum()
    if negative_values_count > 0:
        print(
            f"Warning: {negative_values_count} rows will be dropped because they contain values below 0 in the 'days' column.")
        df = df[df['days'] >= 0]

    return df
