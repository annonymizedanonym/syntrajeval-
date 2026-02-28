import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd




def average_speed_distribution(df: pd.DataFrame):
    df_copy = df.copy()
    try:
        df_copy["time_diff_s"] = (
            df_copy.groupby("tid")["datetime"].diff().dt.total_seconds()
        )
    except KeyError:
        warnings.warn(
            "A column for the temporal attribute was not recognized."
            " If it exists, enter the column's name in TrajectoryDataset."
            " Otherwise, the code will not break but the results will be None."
        )
        return None
    g = df.groupby("tid")
    # Calculate distance between consecutive points within each trajectory
    time_diff = g["datetime"].diff().dt.total_seconds().values
    dx = g["x_m"].diff().values
    dy = g["y_m"].diff().values
    dist = np.sqrt(dx**2 + dy**2)
    mask = (time_diff > 0) & (~np.isnan(dist))
    valid_speeds = (dist[mask] / time_diff[mask]) * 3.6
    return pd.Series(valid_speeds, index=df["tid"][mask]).groupby(level=0).mean().values


def transition_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, List, dict]:
    """
    Build a global first-order transition count matrix from trajectory data.

    The function aggregates transitions across all trajectories to estimate
    how population-level spatial mass moves between locations from one
    observation to subsequent observed locations across trajectories.
    It returns raw transition counts (not probabilities), which allows
    downstream evaluation methods to apply appropriate normalization and
    weighting.

    Note:
    The output is transitions between observations, not transitions over time.

    Returns
    -------
    transition_counts : np.ndarray, shape (n_states, n_states)
        Raw counts of observed transitions from state i to state j.
    states : list of int
        Sorted list of all states appearing in the data.
        Matrix indices correspond to this ordering.
    state_freq : dict
        Dictionary mapping each state to the number of times it appears
        as a transition origin (used for weighting during evaluation).
    """
    df_copy = df[["tid", "cell_idx"]].copy()

    df_copy["s_to"] = df_copy.groupby("tid")["cell_idx"].shift(-1)
    df_trans = df_copy.dropna(subset=["s_to"])

    s_from_vals = df_trans["cell_idx"].values
    s_to_vals = df_trans["s_to"].values

    # Build the matrix
    states_array = np.unique(np.concatenate([s_from_vals, s_to_vals]))
    states_list = states_array.tolist()
    n_states = len(states_array)
    i_indices = np.searchsorted(states_array, s_from_vals)
    j_indices = np.searchsorted(states_array, s_to_vals)

    # Vectorized matrix population
    matrix = np.zeros((n_states, n_states))
    np.add.at(matrix, (i_indices, j_indices), 1.0)

    # Calculate state frequencies using sum over rows
    freq_counts = matrix.sum(axis=1)
    state_freq = dict(zip(states_list, freq_counts))

    return matrix, states_list, state_freq


def reachability(df, max_speed=100):
    def compute_segment_speed(df):
        df = df.sort_values("datetime").copy()
        df = df.drop_duplicates(subset=["tid", "datetime"])

        # Vectorized shifts:
        # differences between consecutive points (diff = point[i+1] - point[i])
        dx = np.diff(df["x_m"].values)
        dy = np.diff(df["y_m"].values)
        dt = np.diff(df["datetime"].values).astype("timedelta64[s]").astype(float)

        # Calculate Euclidean distance: sqrt(dx^2 + dy^2)
        # Pad with NaN at the end to maintain the original dataframe length
        dist_m = np.sqrt(dx**2 + dy**2)
        dist_m = np.append(dist_m, np.nan)
        dt_s = np.append(dt, np.nan)

        # Handle division by zero if two points have the same timestamp
        with np.errstate(divide="ignore", invalid="ignore"):
            speed_kmh = (dist_m / dt_s) * 3.6

        df["speed_kmh"] = speed_kmh
        return df

    try:
        df_reach = df.groupby(["tid"], group_keys=False).apply(compute_segment_speed)

    except KeyError:
        warnings.warn(
            "A column for the temporal attribute was not recognized."
            "If it exists, enter the column's name when defining the TrajectoryDataset."
            "Otherwise, the code will not break but the results will be N/A."
        )
        return "N/A"
    # This removes the last point of every trajectory from consideration
    df_reach.dropna(subset=["speed_kmh"], inplace=True)
    valid_mask = (df_reach["speed_kmh"] >= 0) & (df_reach["speed_kmh"] <= max_speed)
    df_reach["is_speed_outlier"] = ~valid_mask
    teleportation_rate = df_reach["is_speed_outlier"].mean()
    return teleportation_rate
