import pandas as pd

def extract_date_features(df, date_column):
    """
    Extract day, month, and year as features from a date column in a DataFrame.

    Parameters:
    - df: Pandas DataFrame
    - date_column: Name of the date column in the DataFrame

    Returns:
    - DataFrame with additional columns for day, month, and year
    """
    # Convert the date column to datetime format
    df[date_column] = pd.to_datetime(df[date_column])

    # Extract day, month, and year features
    df['day'] = df[date_column].dt.day
    df['month'] = df[date_column].dt.month
    df['year'] = df[date_column].dt.year

    return df


def preprocess_df(df, date_col='Week_Start', target_col="quantity"):

    if target_col in df.columns:
        aggregated_dict = {'quantity': 'sum', 'unit_price': 'max'}
    
        processed_df = df.groupby(["pizza_id", date_col], as_index=False).agg(aggregated_dict)

        processed_df = processed_df.drop(target_col, axis=1)
    else:
        aggregated_dict = {'unit_price': 'max'}
    
        processed_df = df.groupby(["pizza_id", date_col], as_index=False).agg(aggregated_dict)

    # extract day, month and year
    processed_df = extract_date_features(processed_df, date_col)

    processed_df = processed_df.drop(date_col, axis=1)
    return processed_df

