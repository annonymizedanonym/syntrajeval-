import warnings
from typing import Dict, List, Tuple

import numpy as np
import ot

from syntraj_utileval.datasets.dataset import TrajectoryDataset
from syntraj_utileval.stats.spatio_temporal import transition_matrix


# from giddy.ergodic import steady_state, mfpt
def global_flow_prediction(
    original: TrajectoryDataset,
    syn: TrajectoryDataset,
    original_transitions: Tuple[
        np.ndarray, List, Dict
    ],  # to avoid calculating it with every dataset,
    normalize_0_1: float = False,
) -> Dict[str, float]:
    """
    Evaluates generative utility via a Sequential Global Flow Prediction task.

    This function treats the synthetic transition matrix as a linear operator
    and applies it to real population distributions to measure predictive
    fidelity using the Wasserstein (Earth Mover's) Distance.

    Protocol: Train on Synthetic, Test on Real (TSTR).
    """
    (_, states_list_orig, _) = original_transitions

    trans_matrix_syn, states_list_syn, _ = transition_matrix(syn.df_discretized)

    # Define a consistent State Space (The Union)
    states_union = sorted(set(states_list_orig) | set(states_list_syn))
    state_syn_to_idx = {s: i for i, s in enumerate(states_union)}
    n_states = len(states_union)

    if n_states == 0:
        warnings.warn("No mass in both transition matrices.")
        return np.nan

    def _lift_matrix(P, states, state_to_idx, n_states):
        lifted = np.zeros((n_states, n_states))
        # Map old indices to new indices
        new_indices = np.array([state_to_idx[s] for s in states])
        lifted[np.ix_(new_indices, new_indices)] = P
        return lifted

    trans_matrix_syn = _lift_matrix(
        trans_matrix_syn, states_list_syn, state_syn_to_idx, n_states
    )
    # Normalize to Probabilities (MLE)
    row_sums = trans_matrix_syn.sum(axis=1)
    # Use np.errstate to handle divide-by-zero for cells with no outgoing transitions
    with np.errstate(divide="ignore", invalid="ignore"):
        p_matrix_syn = trans_matrix_syn / row_sums[:, np.newaxis]
        p_matrix_syn = np.nan_to_num(p_matrix_syn)  # Replace NaNs with 0

    # Handle "Sink State" problem:
    # If a cell has no outgoing transition (any row that sums to 0.), distribute mass uniformly across the union domain
    p_matrix_syn[row_sums == 0] = 1.0 / n_states

    # Pre-compute Spatial Ground Cost (Meters)
    cell_centroids = {}
    cell_centroids.update(original.cell_centroids)
    cell_centroids.update(syn.cell_centroids)
    # Cell centroids in meters (shared grid assumed)
    coords = np.array(
        [
            cell_centroids[s]  # (x_m, y_m)
            for s in states_union
        ]
    )

    C_full = ot.dist(coords, coords, metric="euclidean")
    if normalize_0_1:
        C_full = C_full / C_full.max() if C_full.max() > 0 else C_full

    # sequential evaluation
    df_orig = original.df_discretized.copy()
    df_orig["step"] = df_orig.groupby("tid").cumcount()
    max_steps = int(
        df_orig["step"].quantile(0.9)
    )  # Avoid tail noise from short trajectories

    # Spatial ground cost

    errors_wd = []

    def _build_step_distributions(df, state_to_idx, n_states, max_steps):
        """
        Builds a (max_steps, n_states) matrix where each row is the distribution
        of the population at that time step.
        """
        # Filter for relevant steps and states
        valid_df = df[df["step"] < max_steps].copy()
        valid_df["state_idx"] = valid_df["cell_idx"].map(state_to_idx)

        # Drop rows where state is not in the union (should be rare)
        valid_df = valid_df.dropna(subset=["state_idx"])

        # Pivot to get counts per step, per state
        counts = valid_df.groupby(["step", "state_idx"]).size().unstack(fill_value=0)

        # Reindex to ensure all states and steps are represented
        full_counts = np.zeros((max_steps, n_states))
        # Use np.ix_ to map the intersection of the found steps and found states
        row_indices = counts.index.values.astype(int)
        col_indices = counts.columns.values.astype(int)
        full_counts[np.ix_(row_indices, col_indices)] = counts.values
        return full_counts

    # vectorize extracting Vn and V_next
    V_all = _build_step_distributions(
        df_orig, state_syn_to_idx, n_states, max_steps + 1
    )
    V_n = V_all[:-1]  # Step 0 to T-1
    V_next = V_all[1:]  # Step 1 to T
    # THE PREDICTION
    V_pred = V_n @ p_matrix_syn

    # Normalize and Calculate Error (MAE)
    for i in range(len(V_pred)):
        if V_pred[i].sum() > 0 and V_next[i].sum() > 0:
            v_p = V_pred[i] / V_pred[i].sum()
            v_true = V_next[i] / V_next[i].sum()
            if v_true.shape[0] > 30000:
                wd = ot.emd2(v_true, v_p, C_full, numItermax=300000)
            else:
                wd = ot.emd2(v_true, v_p, C_full, numItermax=100000)

            errors_wd.append(wd)

        if not errors_wd:
            return {"Sequential_Flow_WD": np.nan, "Step_Degradation": np.nan}

    # Step Degradation: The slope of the error over time
    slope = (
        np.polyfit(range(len(errors_wd)), errors_wd, 1)[0]
        if len(errors_wd) > 1
        else 0.0
    )
    return {"Sequential_Flow_WD": np.mean(errors_wd), "Step_Degradation": float(slope)}


def trajectory_next_location_prediction(
    original: TrajectoryDataset,
    syn: TrajectoryDataset,
    original_transitions: Tuple[np.ndarray, List, Dict],
    normalize_0_1: float = False,
    k: int = 5,
) -> Dict[str, float]:
    """
    Implements Individual MMC Prediction logic (Gambs et al., 2012).
    Measures how well the Synthetic TM predicts the next step
    for individual real trajectories.

    Protocol: Train on Synthetic, Test on Real (TSTR).
    """
    (_, states_list_orig, _) = original_transitions

    trans_matrix_syn, states_list_syn, _ = transition_matrix(syn.df_discretized)

    # Define a consistent State Space (The Union)
    states_union = sorted(set(states_list_orig) | set(states_list_syn))
    state_syn_to_idx = {s: i for i, s in enumerate(states_union)}
    n_states = len(states_union)

    if n_states == 0:
        warnings.warn("No mass in both transition matrices.")
        return np.nan

    def _lift_matrix(P, states, state_to_idx, n_states):
        lifted = np.zeros((n_states, n_states))
        # Map old indices to new indices
        new_indices = np.array([state_to_idx[s] for s in states])
        lifted[np.ix_(new_indices, new_indices)] = P
        return lifted

    trans_matrix_syn = _lift_matrix(
        trans_matrix_syn,
        states_list_syn,
        state_syn_to_idx,
        n_states,
    )

    # Normalize to Probabilities (MLE)
    row_sums = trans_matrix_syn.sum(axis=1)
    # Use np.errstate to handle divide-by-zero for cells with no outgoing transitions
    with np.errstate(divide="ignore", invalid="ignore"):
        p_matrix_syn = trans_matrix_syn / row_sums[:, np.newaxis]
        p_matrix_syn = np.nan_to_num(p_matrix_syn)  # Replace NaNs with 0

    # Handle "Sink State" problem:
    # If a cell has no outgoing transition (any row that sums to 0.), distribute mass uniformly across the union domain
    p_matrix_syn[row_sums == 0] = 1.0 / n_states

    # Pre-compute Spatial Ground Cost (Meters)
    cell_centroids = {}
    cell_centroids.update(original.cell_centroids)
    cell_centroids.update(syn.cell_centroids)
    # Cell centroids in meters (shared grid assumed)
    coords = np.array(
        [
            cell_centroids[s]  # (x_m, y_m)
            for s in states_union
        ]
    )

    C_full = ot.dist(coords, coords, metric="euclidean")
    if normalize_0_1:
        C_full = C_full / C_full.max() if C_full.max() > 0 else C_full

    # sequential evaluation
    df_orig = original.df_discretized.sort_index()
    df_orig["step"] = df_orig.groupby("tid").cumcount()

    # Spatial ground cost

    individual_accuracies = []

    # Iterate through Real Trajectories one by one
    for tid, group in df_orig.groupby("tid"):
        path = group.sort_values("step")["cell_idx"].values
        if len(path) < 2:
            continue

        hits = 0
        total_attempts = 0

        # At each step n, predict n+1
        for step in range(len(path) - 1):
            # Current Mass (Step n)
            curr_state = path[step]
            next_state = path[step + 1]  # ground truth to compare with

            if curr_state in state_syn_to_idx:
                idx = state_syn_to_idx[curr_state]
                # Predict the state with the k hignest probability in the Transition Matrix
                # (Maximum Likelihood Prediction)
                # Get the row of probabilities for the current state
                row_probs = p_matrix_syn[idx]

                # argpartition doesn't sort, but it puts the k largest values at the end
                if len(row_probs) >= k:
                    top_k_indices = np.argpartition(row_probs, -k)[-k:]
                    top_k_states = [states_union[i] for i in top_k_indices]
                else:
                    # Fallback if state space is smaller than K
                    top_k_states = states_union

                if next_state in top_k_states:
                    hits += 1
                total_attempts += 1

        if total_attempts > 0:
            individual_accuracies.append(hits / total_attempts)

    return {
        "mean topk accuracy": np.mean(individual_accuracies),
        "variance topk accuracy": np.var(individual_accuracies),
    }
