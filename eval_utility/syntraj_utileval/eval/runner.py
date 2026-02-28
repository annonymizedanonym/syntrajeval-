import warnings
from typing import TYPE_CHECKING, List, Tuple

import networkx as nx
import numpy as np
import ot
from scipy.stats import kendalltau, wasserstein_distance

from syntraj_utileval.realism import spatial as realism_spatial
from syntraj_utileval.stats import semantic, spatial, spatio_temporal

if TYPE_CHECKING:
    from syntraj_utileval.utils.dataset import TrajectoryDataset


def trajectory_length_error(
    original: "TrajectoryDataset",
    syn: "TrajectoryDataset",
):
    lengths_orig = spatial.trajectory_length_distribution(original.df_discretized)
    lengths_syn = spatial.trajectory_length_distribution(syn.df_discretized)
    return wasserstein_distance(lengths_orig, lengths_syn)  # in kilometers


def density_error(
    original: "TrajectoryDataset",
    syn: "TrajectoryDataset",
    x_bins,
    y_bins,
):
    union_bbox_m = syn.union_bbox_m
    D_original, X, Y = spatial.make_density_grid_from_df(
        original.df_discretized,
        union_bbox_m,
        x_bins=x_bins,
        y_bins=y_bins,
    )
    D_syn, X, Y = spatial.make_density_grid_from_df(
        syn.df_discretized,
        union_bbox_m,
        x_bins=x_bins,
        y_bins=y_bins,
    )

    return spatial.compute_2d_wasserstein(D_original, D_syn, X, Y)


def origin_density_error(
    original: "TrajectoryDataset",
    syn: "TrajectoryDataset",
    x_bins,
    y_bins,
):
    union_bbox_m = syn.union_bbox_m

    D_original, X, Y = spatial.make_density_grid_from_df(
        original.df_discretized.groupby("tid").first().reset_index(),
        union_bbox_m,
        x_bins=x_bins,
        y_bins=y_bins,
    )
    D_syn, X, Y = spatial.make_density_grid_from_df(
        syn.df_discretized.groupby("tid").first().reset_index(),
        union_bbox_m,
        x_bins=x_bins,
        y_bins=y_bins,
    )

    return spatial.compute_2d_wasserstein(D_original, D_syn, X, Y)


def i_rank_similarity(original: "TrajectoryDataset", syn: "TrajectoryDataset") -> float:
    """
    Compare individual rank (i-rank) between the original
    and synthetic dataset using a projected I-rank metric
    The evaluation proceeds in three steps:

    1. Compute per-user rank statistics (location visit ranks per user).
    2. Project each user's rank vector into a 1D scalar using kendall-tau with a refernce T-rank (cell_idx sorted)
    3. Compare the resulting scalar distributions using the 1D Wasserstein
    distance (Earth Mover's Distance).

    Parameters
    ----------
    original : Dataset
        Dataset containing the original (real) trajectories, already
        discretized onto a shared spatial grid.
    syn : Dataset
        Dataset containing the synthetic trajectories, discretized using
        the same grid definition.

    Returns
    -------
    float
        Wasserstein-1 distance between the distributions of per-user
        projected I-rank scores.
    """
    ir_orig = spatial.i_rank(original.df_discretized)
    ir_syn = spatial.i_rank(syn.df_discretized)

    def rank_to_tau(i_r):
        rank_scores = []
        for tid, rank in i_r.groupby("tid"):
            rank = rank.sort_values("cell_idx")["rank"].to_numpy()

            n = len(rank)

            if n < 2:
                tau = 0.5
            else:
                tau, _ = kendalltau(rank, np.arange(len(rank)), variant="b")
                if np.isnan(tau):
                    # all values are tied
                    tau = 0.0
            # rank_scores.append((1 - tau) / 2)
            rank_scores.append(tau)

        return rank_scores

    ir_orig_scores = np.array(rank_to_tau(ir_orig))
    ir_syn_scores = np.array(rank_to_tau(ir_syn))

    # return distributional comparison
    return ot.wasserstein_1d(ir_orig_scores, ir_syn_scores)


def G_rank_categorical_similarity(
    original: "TrajectoryDataset", syn: "TrajectoryDataset", n_cat: int
) -> float:
    gr_cat_orig = semantic.g_rank_category(original.df, n_cat)
    gr_cat_syn = semantic.g_rank_category(syn.df, n_cat)
    # distance = (1 - corr) / 2

    if gr_cat_syn is None or gr_cat_orig is None:
        return "N/A"
    else:
        tau = kendalltau(gr_cat_orig, gr_cat_syn, variant="b")
        return tau.statistic


def G_rank_similarity(original: "TrajectoryDataset", syn: "TrajectoryDataset") -> float:
    union_cells = sorted(
        set(original.df_discretized["cell_idx"]).union(syn.df_discretized["cell_idx"])
    )
    gr_orig = spatial.g_rank(original.df_discretized, union_cells)
    gr_syn = spatial.g_rank(syn.df_discretized, union_cells)

    tau = kendalltau(gr_orig, gr_syn, variant="b")
    return tau.statistic


def average_speed_distribution_similarity(
    original: "TrajectoryDataset", syn: "TrajectoryDataset"
) -> float:
    mean_avg_speed_orig = spatio_temporal.average_speed_distribution(
        original.df_discretized
    )
    mean_avg_speed_syn = spatio_temporal.average_speed_distribution(syn.df_discretized)

    if mean_avg_speed_orig is None or mean_avg_speed_syn is None:
        return "N/A"

    return ot.wasserstein_1d(mean_avg_speed_orig, mean_avg_speed_syn)


def transition_similarity_rowwise(
    original: "TrajectoryDataset",
    syn: "TrajectoryDataset",
    original_transitions: Tuple[
        np.ndarray, List, dict
    ],  # to avoid calculating it with every dataset
    normalize_0_1,
) -> float:
    """
    Compare global first-order transition matrices using a weighted row-wise wasserstein distance with a spatial ground cost.
    Given that a location is observed, how does the distribution of next observed locations differ between datasets
    For each origin state i, the method compares the conditional
    distributions of next observed locations.

    These per-state distances are aggregated using visitation frequencies
    from the original dataset as weights.

    Note:
    -This metric assumes first-order Morkov structure and population-level Markov kernel comparison

    Returns
    -------
    float
        Weighted row-wise Wasserstein distance between transition matrices.
        Lower values indicate higher similarity of transition behavior.


    """

    # Row-stochastic transition matrix of the original dataset.
    # Rows with zero outgoing transitions are allowed.
    (P_orig, states_orig, freq_orig) = original_transitions

    P_syn, states_syn, freq_syn = spatio_temporal.transition_matrix(syn.df_discretized)

    #  empirical union support
    freq_orig = dict(freq_orig)
    freq_syn = dict(freq_syn)
    states_union = sorted(set(states_orig) | set(states_syn))
    n_states = len(states_union)

    if n_states == 0:
        warnings.warn("No mass in both transition matrices.")
        return np.nan

    # Adapt both transition matrices to the union support
    state_to_idx = {s: i for i, s in enumerate(states_union)}


    def _lift_matrix(P, states, state_to_idx, n_states):
        lifted = np.zeros((n_states, n_states))
        # Map old indices to new indices
        new_indices = np.array([state_to_idx[s] for s in states])
        lifted[np.ix_(new_indices, new_indices)] = P
        return lifted

    P_orig_u = _lift_matrix(P_orig, states_orig, state_to_idx, n_states)
    P_syn_u = _lift_matrix(P_syn, states_syn, state_to_idx, n_states)
    # Weights with original as reference
    weights = np.zeros(n_states)
    total_freq = sum(freq_orig.values())

    for s, f in freq_orig.items():
        weights[state_to_idx[s]] = (
            f / total_freq
        )  # Normalize state frequencies to weights

    # Spatial ground cost
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
        C_full = C_full / C_full.max()

    # weight the row-wise Wasserstein
    total_distance = 0.0
    eps = 1e-12
    for i in range(n_states):
        w_i = weights[i]
        if w_i < eps:
            continue  # skip nearly empty row

        p = P_orig_u[i]
        q = P_syn_u[i]

        # reduce to support by row of both transition matrices
        support = np.where((p > 0) | (q > 0))[0]
        if support.size == 0:
            continue

        p_s = p[support]
        q_s = q[support]

        # Ensure we have mass in the original row to compare
        if p.sum() == 0:
            # This state exists in the union but not in our original reference
            # Since we weight by original frequencies, this shouldn't happen,
            # but we skip it for mathematical rigor.
            continue

        elif p.sum() > 0 and q.sum() == 0:
            # The model never visited this cell as a 'from' state
            # We penalize this as a maximum spatial mismatch
            # this already covered in the compute_2d_wasserstein function
            total_distance += w_i * 1.0
            continue
        else:
            p_s = p_s / (p_s.sum())
            q_s = q_s / (q_s.sum())
            C = C_full[np.ix_(support, support)]

            d_i = ot.emd2(p_s, q_s, C)
            total_distance += w_i * d_i

    return float(total_distance)


def i_mean_rank_displacement(
    original: "TrajectoryDataset",
    syn: "TrajectoryDataset",
    N: int,
    metric_crs,
    cell_width,
    cell_height,
    union_bbox_m,
    missing_penalty=None,
):
    """
    Mean I-rank displacement at top-N, averaged over individuals.
    """

    ir_orig = spatial.i_rank(original.df_discretized)
    ir_syn = spatial.i_rank(syn.df_discretized)

    users = set(ir_orig["tid"].unique()) & set(ir_syn["tid"].unique())
    nmrds = []

    for uid in users:
        orig_u = ir_orig[ir_orig["tid"] == uid].sort_values("rank").head(N)
        syn_u = ir_syn[ir_syn["tid"] == uid]

        rank_syn = {
            row["cell_idx"]: int(row["rank"]) - 1 for _, row in syn_u.iterrows()
        }

        missing_penalty = len(orig_u)

        displacements = [
            abs(i - min(rank_syn.get(cell, missing_penalty), missing_penalty))
            for i, cell in enumerate(orig_u["cell_idx"].values)
        ]

        mrd_u = np.mean(displacements)

        # worst-case average displacement for this user
        worst_u = np.mean([abs(i - missing_penalty) for i in range(len(orig_u))])
        nmrds.append(mrd_u / worst_u if worst_u > 0 else 0.0)

    return float(np.mean(nmrds))


def category_loc_top1_match(
    original: "TrajectoryDataset",
    syn: "TrajectoryDataset",
    min_support=5,
    dominance=0.5,
):
    category_loc_count_original = semantic.category_loc_counts(original.df_discretized)

    category_loc_count_syn = semantic.category_loc_counts(syn.df_discretized)

    if category_loc_count_original is None or category_loc_count_syn is None:
        return {"match_rate": "N/A", "coverage": "N/A"}
    cats = sorted(
        set(category_loc_count_original.columns) | set(category_loc_count_syn.columns)
    )

    category_loc_count_original = category_loc_count_original.reindex(
        columns=cats, fill_value=0
    )
    category_loc_count_syn = category_loc_count_syn.reindex(columns=cats, fill_value=0)

    # Support and top-1 in original
    support_orig = category_loc_count_original.sum(axis=1)
    support_syn = category_loc_count_syn.sum(axis=1)

    top1_idx = category_loc_count_original.values.argmax(axis=1)
    top1_cat = np.array(cats, dtype=object)[top1_idx]
    top1_share = category_loc_count_original.values.max(axis=1) / np.maximum(
        support_orig.values, 1
    )

    # Eligibility by original data
    eligible = (support_orig >= min_support) & (top1_share >= dominance)
    if not eligible.any():
        warnings.warn("No cells are eligible, check input data or reduce min_support")
        return {
            "match_rate": np.nan,
            "match_rate_weighted": np.nan,
            "n_cells": 0,
            "coverage": 0.0,
        }

    # Compute top-1 in syn, aligned to the same cell index set (original is reference)
    category_loc_count_syn = category_loc_count_syn.reindex(
        index=category_loc_count_original.index, fill_value=0
    )
    top1_idx_syn = category_loc_count_syn.values.argmax(axis=1)
    top1_cat_syn = np.array(cats, dtype=object)[top1_idx_syn]

    # Compare only eligible cells
    idx = np.where(eligible.values)[0]
    matches = (top1_cat[idx] == top1_cat_syn[idx]) & (support_syn.iloc[idx].values > 0)

    match_rate = float(matches.mean())
    coverage = float(eligible.mean())

    return {"match_rate": match_rate, "coverage": coverage}


def pairwise_distance_distribution_similiarity(
    original: "TrajectoryDataset",
    syn: "TrajectoryDataset",
    distance="frechet",
    normalize_0_1=False,
    n_jobs=1,
):
    if distance == "frechet":
        original_f_distance_mat = spatial.pairwise_frechet(
            original.df_discretized, n_jobs
        )
        original_triu_indices = np.triu_indices_from(original_f_distance_mat, k=1)
        original_pairwise_dist = original_f_distance_mat[original_triu_indices]

        syn_f_distance_mat = spatial.pairwise_frechet(syn.df_discretized, n_jobs)
        syn_triu_indices = np.triu_indices_from(syn_f_distance_mat, k=1)
        syn_pairwise_dist = syn_f_distance_mat[syn_triu_indices]

        if normalize_0_1:
            return (
                ot.wasserstein_1d(original_pairwise_dist, syn_pairwise_dist)
                / original_pairwise_dist.max()
            )

        else:
            return ot.wasserstein_1d(original_pairwise_dist, syn_pairwise_dist)

    if distance == "hausdorff":
        original_haus_distance_mat = spatial.pairwise_hausdorff(
            original.df_discretized, n_jobs
        )
        original_triu_indices = np.triu_indices_from(original_haus_distance_mat, k=1)
        original_pairwise_dist = original_haus_distance_mat[original_triu_indices]

        syn_haus_distance_mat = spatial.pairwise_hausdorff(syn.df_discretized, n_jobs)
        syn_triu_indices = np.triu_indices_from(syn_haus_distance_mat, k=1)
        syn_pairwise_dist = syn_haus_distance_mat[syn_triu_indices]

        if normalize_0_1:
            return (
                ot.wasserstein_1d(original_pairwise_dist, syn_pairwise_dist)
                / original_pairwise_dist.max()
            )

        else:
            return ot.wasserstein_1d(original_pairwise_dist, syn_pairwise_dist)

    if distance == "cosine_category":
        try:
            original_cosine_distance_mat = semantic.pairwise_cosine(original.df, n_jobs)
            original_triu_indices = np.triu_indices_from(
                original_cosine_distance_mat, k=1
            )
            original_pairwise_dist = original_cosine_distance_mat[original_triu_indices]

            syn_cosine_distance_mat = semantic.pairwise_cosine(syn.df, n_jobs)
            syn_triu_indices = np.triu_indices_from(syn_cosine_distance_mat, k=1)
            syn_pairwise_dist = syn_cosine_distance_mat[syn_triu_indices]
        except KeyError:
            warnings.warn(
                f"A column for the 'category' was not recognized for the {syn.dataset_name} dataset."
                " If it exists, enter the column's name in TrajectoryDataset."
                " Otherwise, the code will not break but the results will be None."
            )
            return "N/A"

        return ot.wasserstein_1d(original_pairwise_dist, syn_pairwise_dist)