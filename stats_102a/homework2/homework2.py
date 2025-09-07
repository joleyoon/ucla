import pandas as pd
import numpy as np


def messy_impute(data, center="min", margin=None, **kwargs):
    """
    Imputes missing values in a messy (wide-format) gradebook.

    Args:
        data (pd.DataFrame): DataFrame with student IDs in the first column and assignment scores in the rest.
        center (str): Statistic for imputation ("mean", "median", "min").
        margin (int): 1 = row-wise imputation, 2 = column-wise imputation.
        **kwargs: Additional arguments passed to aggregation functions (e.g., skipna=True).

    Returns:
        pd.DataFrame: DataFrame with missing values imputed.
    """
    if center not in ["mean", "median", "min"]:
        raise ValueError("Invalid center, must be 'mean', 'median', or 'min'")
    if margin not in [1, 2]:
        raise ValueError("Invalid margin, must be 1 or 2")

    df = data.copy()
    no_uid = df.iloc[:, 1:].copy()

    if no_uid.notna().all().all():
        return df

    fun = {
        "mean": lambda x: np.nanmean(x, **kwargs),
        "median": lambda x: np.nanmedian(x, **kwargs),
        "min": lambda x: np.nanmin(x, **kwargs),
    }

    na_indices = np.argwhere(pd.isna(no_uid).values)

    if margin == 1:  # row-wise
        for row_idx, col_idx in na_indices:
            row_vals = no_uid.iloc[row_idx, :].values
            impute_val = fun[center](row_vals)
            no_uid.iat[row_idx, col_idx] = int(impute_val)
    elif margin == 2:  # column-wise
        for row_idx, col_idx in na_indices:
            col_vals = no_uid.iloc[:, col_idx].values
            impute_val = fun[center](col_vals)
            no_uid.iat[row_idx, col_idx] = int(impute_val)

    df.iloc[:, 1:] = no_uid
    return df


def tidy_impute(data, center="min", margin=None, **kwargs):
    """
    Imputes missing values in a tidy (long-format) gradebook.

    Args:
        data (pd.DataFrame): DataFrame with columns ["UID", "Assignment", "Score"].
        center (str): Statistic for imputation ("mean", "median", "min").
        margin (int): 1 = student-wise imputation, 2 = assignment-wise imputation.
        **kwargs: Additional arguments passed to aggregation functions.

    Returns:
        pd.DataFrame: DataFrame with missing values imputed.
    """
    if center not in ["mean", "median", "min"]:
        raise ValueError("Invalid center, must be 'mean', 'median', or 'min'")
    if margin not in [1, 2]:
        raise ValueError("Invalid margin, must be 1 or 2")

    fun = {
        "mean": lambda x: np.nanmean(x, **kwargs),
        "median": lambda x: np.nanmedian(x, **kwargs),
        "min": lambda x: np.nanmin(x, **kwargs),
    }

    df = data.copy()

    if margin == 1:  # group by student
        df["Score"] = df.groupby("UID")["Score"].transform(
            lambda s: s.fillna(int(fun[center](s.values)))
        )
    elif margin == 2:  # group by assignment
        df["Score"] = df.groupby("Assignment")["Score"].transform(
            lambda s: s.fillna(int(fun[center](s.values)))
        )

    return df