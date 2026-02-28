import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial.distance import cosine


def category_loc_counts(
    df: pd.DataFrame,
):
    """
    Check top-1 category to cell_idx match in df2 is the same df1
    - Computes the dominant (top-1) category per cell in df1 (reference)
    - restricted to cells with enough support and clear dominance.

    Returns
        match_rate: fraction of eligible cells whose dominant category matches in synthetic
        coverage: fraction of all cells that are eligible (by real data)

    Notes
    - If no cells are eligible, returns match_rate = NaN and coverage = 0.0.
    """
    try:
        df_ = df[["cell_idx", "category"]].copy()
    except KeyError:
        return None

    # Counting occurrences of combinations of values.
    ct = pd.crosstab(df_["cell_idx"], df_["category"])

    return ct


def g_rank_category(df, n_cat):
    try:
        g_rank = (
            df.groupby("category").size().sort_values(ascending=False).index.tolist()
        )
    except KeyError:
        warnings.warn(
            "A column for the category was not recognized."
            " If it exists, enter the column's name in TrajectoryDataset."
            " Otherwise, the code will not break but the results will be None."
        )
        return None

    if n_cat == len(g_rank):
        return g_rank
    else:
        missing = list(set(np.arange(n_cat).tolist()) - set(g_rank))
        g_rank = g_rank + missing
        return g_rank


def pairwise_cosine(df, normalize_0_1=False, n_jobs=1):
    def compute_cosine(i, j, trajectory1, trajectory2):
        n1 = len(trajectory1)
        n2 = len(trajectory2)

        if n1 < n2:
            trajectory2 = trajectory2[:n1]
        else:
            trajectory1 = trajectory1[:n2]

        if normalize_0_1:
            return (
                cosine(trajectory1, trajectory2) / 2
            )  # divide by two to normalize range [0,1]
        else:
            return cosine(trajectory1, trajectory2)

    traj_array = []  # array(n,2)
    for tid, traj in df.groupby("tid")["category"]:
        traj_array.append(traj)

    return Parallel_computing(compute_cosine, traj_array, n_jobs)


def Parallel_computing(function, array, n_jobs=1):
    results = Parallel(n_jobs=n_jobs)(
        delayed(function)(i, j, array[i], array[j])
        for i in range(len(array))
        for j in range(i, len(array))
    )

    n_trajectories = len(array)
    dist_matrix = np.zeros((n_trajectories, n_trajectories))

    idx = 0

    for i in range(n_trajectories):
        for j in range(i, n_trajectories):
            dist_matrix[i, j] = results[idx]
            dist_matrix[j, i] = dist_matrix[i, j]
            idx += 1
    return dist_matrix
