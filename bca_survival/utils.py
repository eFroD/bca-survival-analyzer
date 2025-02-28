import pandas as pd
import numpy as np
from datetime import datetime
def make_quantile_split(df, column):
    """
    Splits a DataFrame column into quantile-based groups ("low", "high", or missing).

    This function computes the 25th (Q1) and 75th (Q3) percentiles for the specified column
    and assigns each row to one of three groups based on the value in the column:
    - "low" for values less than or equal to Q1.
    - "high" for values greater than or equal to Q3.
    - Missing (pd.NA) for values between Q1 and Q3.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data to split.
        column (str): The name of the column to be analyzed and split into quantile-based groups.

    Returns:
        pd.DataFrame: A new DataFrame with an additional "group" column that indicates the assigned group.
                      Rows with missing values in the specified column are excluded from the result.

    Example:
        >>> import pandas as pd
        >>> data = {'values': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
        >>> df = pd.DataFrame(data)
        >>> result = make_quantile_split(df, column="values")
        >>> print(result)

    """
    q25 = df[column].quantile(q=0.25).item()
    q75 = df[column].quantile(q=0.75).item()

    def assign_group(x):
        if x <= q25:
            return "low"
        elif x >= q75:
            return "high"
        else:
            return pd.NA

    df_tmp = df.copy().dropna(subset=column)
    df_tmp["group"] = df_tmp[column].apply(assign_group)
    return df_tmp


def make_quantile_split_outter_vs_middle(df, column):
    """
    Splits a DataFrame column into quantile-based groups ("outter", "middle").

    This function computes the 25th (Q1) and 75th (Q3) percentiles for the specified column
    and assigns each row to one of three groups based on the value in the column:
    - "outter" for values less than or equal to Q1 or greater than or equal to Q3.
    - "middle" for values in between these two quantiles

    Args:
        df (pd.DataFrame): The input DataFrame containing the data to split.
        column (str): The name of the column to be analyzed and split into quantile-based groups.

    Returns:
        pd.DataFrame: A new DataFrame with an additional "group" column that indicates the assigned group.
                      Rows with missing values in the specified column are excluded from the result.

  """
    q25 = df[column].quantile(q=0.25).item()
    q75 = df[column].quantile(q=0.75).item()

    def assign_group(x):
        if x <= q25:
            return "outer"
        elif x >= q75:
            return "outer"
        else:
            return "middle"
    df_tmp = df.copy().dropna(subset=column)
    df_tmp["group"] = df_tmp[column].apply(assign_group)
    return df_tmp


def calculate_age(df, birth_date_col, current_date_col, age_col_name='Age'):
    """
    Calculate age in years from two columns storing dates in a pandas DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the date columns.
        birth_date_col (str): The name of the column with the birth date.
        current_date_col (str): The name of the column with the current/reference date.
        age_col_name (str): The name of the new column to store calculated age. Default is 'Age'.

    Returns:
        pd.DataFrame: DataFrame with the new column for age.
    """
    df[birth_date_col] = pd.to_datetime(df[birth_date_col], format='%d.%m.%Y', errors='coerce')
    df[current_date_col] = pd.to_datetime(df[current_date_col], format='%d.%m.%Y', errors='coerce')

    df[age_col_name] = np.where(
        df[birth_date_col].notna() & df[current_date_col].notna(),
        df[current_date_col].dt.year - df[birth_date_col].dt.year - (
                (df[current_date_col].dt.month < df[birth_date_col].dt.month) |
                ((df[current_date_col].dt.month == df[birth_date_col].dt.month) &
                 (df[current_date_col].dt.day < df[birth_date_col].dt.day))
        ),
        np.nan)
    return df

def clean_dates(df, date_column, date_format=None):
    """
    Cleans a DataFrame by removing rows with invalid date values.

    Parameters:
    df (pandas.DataFrame): The DataFrame to be cleaned.
    date_column (str): The name of the column containing date values.
    date_format (str, optional): The format of the date values (e.g., '%Y-%m-%d').
                                 If None, various formats will be automatically detected.

    Returns:
    pandas.DataFrame: Cleaned DataFrame without invalid date values.
    dict: Statistics about the cleaning process
    """
    original_rows = len(df)
    df_clean = df.copy()

    if date_format is None:
        df_clean[date_column] = pd.to_datetime(df_clean[date_column], errors='coerce')
    else:
        def validate_date(date_str):
            try:
                datetime.strptime(str(date_str), date_format)
                return pd.to_datetime(date_str, format=date_format)
            except (ValueError, TypeError):
                return pd.NaT

        df_clean[date_column] = df_clean[date_column].apply(validate_date)

    df_clean = df_clean.dropna(subset=[date_column])

    removed_rows = original_rows - len(df_clean)
    stats = {
        'original_rows': original_rows,
        'cleaned_rows': len(df_clean),
        'removed_rows': removed_rows,
        'removal_percentage': round((removed_rows / original_rows) * 100, 2)
    }

    return df_clean, stats
