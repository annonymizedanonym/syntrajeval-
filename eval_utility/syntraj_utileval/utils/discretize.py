import numpy as np
import pandas as pd

from syntraj_utileval.datasets.dataset import TrajectoryDataset


def step_length(df, min_move_m=1):
    try:
        df_transition = df.sort_values(["tid", "datetime"])
    except KeyError:
        df_transition = df.reset_index().sort_values(["tid", "index"])

    dx = df_transition.groupby("tid")["centroid_x"].diff()
    dy = df_transition.groupby("tid")["centroid_y"].diff()
    d = np.sqrt(dx * dx + dy * dy)
    d = d[d >= min_move_m]  # remove near-zero moves/stops
    return d.dropna()


def spatial_entropy(TrajDataset: TrajectoryDataset, global_origin, cell_w, cell_h):
    TrajDataset.discretized(cell_w, cell_h, global_origin)

    _, total_cells = snap_bbox(TrajDataset.bbox_m, global_origin, cell_w, cell_h)

    try:
        df_transition = TrajDataset.df_discretized.sort_values(["tid", "datetime"])
    except KeyError:
        df_transition = TrajDataset.df_discretized.reset_index().sort_values(
            ["tid", "index"]
        )

    df_transition["next_c_idx"] = df_transition.groupby("tid")["cell_idx"].shift(-1)
    transitions = df_transition.dropna(subset=["next_c_idx"]).copy()
    total = int(len(transitions))

    # Self-transitions
    self_mask = transitions["cell_idx"] == transitions["next_c_idx"]
    self_count = int(self_mask.sum())
    self_frac = self_count / total

    # Unique non-self transitions (exclude i->i)
    if (~self_mask).any():
        unique_nonself = int(
            transitions.loc[~self_mask].groupby(["cell_idx", "next_c_idx"]).ngroups
        )
    else:
        unique_nonself = 0

    unique_non_self_frac = unique_nonself / total

    # Active-cell coverage: use all visited cells (not just those in transitions)
    visited_cells = pd.Index(
        df_transition["cell_idx"].dropna().astype("int64")
    ).unique()
    K_active = int(visited_cells.size)

    active_cell_frac = K_active / total_cells

    return unique_non_self_frac, self_frac, active_cell_frac


def find_knee(x, y, direction="decreasing", window=3):
    x = np.asarray(x)
    y = np.asarray(y)

    # check validity
    order = np.argsort(x)
    x, y = x[order], y[order]

    dy = np.diff(y)
    d2y = np.diff(dy)

    # 1- check monotonicity
    if direction == "decreasing":
        mono = dy <= 1e-8
    else:
        mono = dy >= -1e-8
    if np.mean(mono) >= 0.8:
        pass
    else:
        return None


    # Normalize
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())

    if direction == "decreasing":
        dist = (1 - x_norm) - y_norm
    else:
        dist = y_norm - x_norm

    return x[np.argmax(dist)]


def snap_bbox(bbox_m, grid_origin, cw, ch):
    """
    snap bounding boxes to a the grid's origin
    """
    xmin, ymin, xmax, ymax = bbox_m
    ox, oy = grid_origin

    j_min = np.floor((xmin - ox) / cw)
    j_max = np.floor((xmax - ox) / cw)
    i_min = np.floor((ymin - oy) / ch)
    i_max = np.floor((ymax - oy) / ch)

    xmin_snap = ox + j_min * cw
    xmax_snap = ox + (j_max + 1) * cw

    ymin_snap = oy + i_min * ch
    ymax_snap = oy + (i_max + 1) * ch
    snaped_bbox_m = xmin_snap, ymin_snap, xmax_snap, ymax_snap
    nx = j_max - j_min + 1
    ny = i_max - i_min + 1

    N_total = nx * ny

    return snaped_bbox_m, N_total
