import pandas as pd


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
