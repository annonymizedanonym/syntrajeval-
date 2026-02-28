import warnings
from collections import defaultdict
from typing import Tuple

import hdbscan
import numpy as np
import ot
import pandas as pd
import shapely
from joblib import Parallel, delayed
from scipy.spatial import cKDTree
from scipy.spatial.distance import directed_hausdorff
from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.neighbors import KernelDensity


def reduce_to_support(
    v1: np.ndarray,
    v2: np.ndarray,
    X: np.ndarray = None,
    Y: np.ndarray = None,
    coords: np.ndarray = None,
    eps_zero: float = 0,
):
    """
    Reduce to union supports S = {i | v1[i] > eps_zero or v2[i] > eps_zero}.
    All inputs are 1D arrays of the same length (num_cells).

    Returns
    -----
    masked arrays.

    """
    mask = (v1 > eps_zero) | (v2 > eps_zero)
    if X and Y:
        return v1[mask], v2[mask], X[mask], Y[mask]
    elif coords is not None:
        return v1[mask], v2[mask], coords[mask]
    else:
        raise ValueError("Either X and Y or coords must be provided.")


def make_density_grid_from_df(
    df: pd.DataFrame,
    bbox_m: Tuple[float, float, float, float],
    x_bins: int = 100,
    y_bins: int = 100,
    smooth: bool = False,
    kde_bandwidth=None,
):
    """
    Calculate a 2D density grid from a DataFrame of points.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing x and y columns (x,y projected latitude and longitude in meter).
    bbox : tuple or None
        Bounding box in meters (min_x, min_y, max_x, max_y). If None, will be computed from data. Bounding box in degrees will distort result
    x_bins : int
        Number of bins along the x (east-west) axis.
    y_bins : int
        Number of bins along the y (north-south) axis.
    smooth : bool
        If True, use kernel density estimation (KDE); otherwise, use a histogram.
    kde_bandwidth : float or None
        Bandwidth for KDE. If None, use default.

    Returns
    -------
    D : np.ndarray
        Density grid (shape: [y_bins, x_bins]).
    X : np.ndarray
        Meshgrid of x (east-west) bin centers.
    Y : np.ndarray
        Meshgrid of y (north-south) bin centers.
    """

    x_m = df["x_m"]
    y_m = df["y_m"]

    min_x, min_y, max_x, max_y = bbox_m

    # Bin edges and centers
    x_edges = np.linspace(min_x, max_x, x_bins + 1)
    y_edges = np.linspace(min_y, max_y, y_bins + 1)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    X, Y = np.meshgrid(
        x_centers, y_centers, indexing="ij"
    )  # Two arrays that contains the center coordinates for x and y

    if smooth:
        grid_points = np.vstack([X.ravel(), Y.ravel()])
        kde = gaussian_kde(np.vstack([x_m, y_m]), kde_bandwidth)
        D = kde(grid_points).reshape(X.shape)

    else:
        D, _, _ = np.histogram2d(
            x_m, y_m, bins=[x_edges, y_edges], range=[[min_x, max_x], [min_y, max_y]]
        )

    return D, X, Y


def trajectory_length_distribution(df):
    df = df.copy()
    try:
        df.sort_values(by=["tid", "datetime"], inplace=True)
    except KeyError:
        df = df.reset_index().sort_values(["tid", "index"])

    diffs_sq = df.groupby("tid")[["x_m", "y_m"]].diff() ** 2
    df["distance_m"] = np.sqrt(diffs_sq["x_m"] + diffs_sq["y_m"])
    lengths = df.groupby("tid")["distance_m"].sum() / 1000.0  # in kilometers
    return lengths


def compute_2d_wasserstein(
    D1: np.ndarray,
    D2: np.ndarray,
    X: np.ndarray = None,
    Y: np.ndarray = None,
    coords: np.ndarray = None,
    eps_zero: float = 0,
    normalize_0_1=False,
    mode_test=False,
    cost_M=None,
) -> float:
    """
    :params D1, D2: w2D np.ndarray of densities or count grids (NOT normalized)
    :params X,Y np.ndarray: meshgrid array of grid cell (encode the coordinates of the grid cells over your geographic area, X: where each element holds the longitude (x-coordinate) of a grid cell center.
                                                                                                                             Y:  where each element holds the latitude (y-coordinate) of a grid cell center.)
    eps_zero: threshol to prune tiny masses (0.0 = exact)

    since EMD only depends on the support (nonzero entries) of the marginals. Any grid cell where D1[i]=0 and D2[i]=0 contributes nothing: its row/column in the cost matrix is never used.
    Therefore, we only apply EMD on the union of supports
    The results is identical to the full-grid problem

    Returns
    -------
    float

        full-support EMD (Wasserstein) distance between D1 and D2

    """
    assert D1.shape == D2.shape, (
        f"D1 and D2 must have the same shape, got {D1.shape} and {D2.shape}"
    )
    d1, d2 = (
        D1.ravel(),
        D2.ravel(),
    )  # Flattens but returns a view and not an independent copy (view == reference to the original) if possible otherwise a copy

    if not mode_test:
        if np.array_equal(d1, d2):
            return 0.0

    if X is not None and Y is not None:
        assert D1.shape == X.shape == Y.shape, (
            f"Shapes of D1, X, and Y must match. Got {D1.shape}, {X.shape}, {Y.shape}"
        )  # Coordinates of grid cell centers, shape (num_cells,2)
        xs, ys = X.ravel(), Y.ravel()

        D1supp, D2supp, Xsupp, Ysupp = reduce_to_support(d1, d2, X=xs, Y=ys)
        coords_supp = np.vstack([Xsupp, Ysupp]).T

    elif coords is not None:
        D1supp, D2supp, coords_supp = reduce_to_support(d1, d2, coords=coords)
    else:
        if cost_M is None:
            raise ValueError(
                "Either X and Y, coords or the ground truth must be provided."
            )

    D1suppsum = D1supp.sum()
    D2suppsum = D2supp.sum()
    if D1suppsum == 0 and D2suppsum == 0:
        warnings.warn("Both the distributions has no mass")
        return np.nan
    elif D1suppsum == 0 or D2suppsum == 0:
        warnings.warn("one of the distributions has no mass")
        return 1.0

    else:
        D1supp /= D1suppsum  # Normalize to sum to 1 (to be able to compute balanced Wasserstein distance), because sometimes D1 and D2 do have matching sums. If m1 ≠ m2, balanced EMD is infeasible (error).
        D2supp /= (
            D2suppsum  # This is isn't interfering with the full-support metric criteria
        )
    # Compute EMD (Wasserstein) emd2 accepts 1D nonnegative histograms with a matching mass sum (Not necessarly normalized)
    if cost_M is None:
        cost_M = ot.dist(
            coords_supp, coords_supp, metric="euclidean"
        )  # compute cost matrix: pairwide distances between all pairwise cell centers
        if normalize_0_1:
            cost_M /= cost_M.max()  # Normalize in [0,1]

    return ot.emd2(D1supp, D2supp, cost_M)


def pairwise_hausdorff(df, n_jobs):
    def compute_directed_hausdorff(i, j, trajectory1, trajectory2):
        return directed_hausdorff(trajectory1, trajectory2)[0]

    traj_array = []
    for tid, traj in df.groupby("tid")[["x_m", "y_m"]]:
        traj_array.append(traj.values)

    return Parallel_computing(compute_directed_hausdorff, traj_array, n_jobs=n_jobs)


def pairwise_frechet(df, n_jobs):
    def compute_frechet(i, j, trajectory1, trajectory2):
        return shapely.frechet_distance(trajectory1, trajectory2)

    traj_array = []  # array(n,2)
    for tid, traj in df.groupby("tid")[["x_m", "y_m"]]:
        traj_array.append(shapely.LineString(traj.values))

    return Parallel_computing(compute_frechet, traj_array, n_jobs=n_jobs)


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


def g_rank(df, all_cells):
    g_rank = (
        df.groupby("cell_idx")
        .size()
        .sort_values(ascending=False)
        .reindex(all_cells, fill_value=0)
        .rank(method="dense", ascending=False)
    )
    return g_rank


def i_rank(df):
    return (
        df.groupby(["tid", "cell_idx"])
        .size()
        .reset_index(name="count")
        .sort_values(["tid", "count"], ascending=[True, False])
        .assign(
            rank=lambda x: x.groupby("tid")["count"].rank(
                method="dense",
                ascending=False,  # dense: ties get the same rank, next rank is consecutive
            )
        )
    )


