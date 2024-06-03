import pandas as pd


def preprocess_data(df, start_date_col, event_date_col, event_col):
    if event_date_col:
        df['days'] = (pd.to_datetime(df[event_date_col]) - pd.to_datetime(df[start_date_col])).dt.days
    df['status'] = df[event_col].astype(int)
    return df
