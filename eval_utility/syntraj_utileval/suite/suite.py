import re
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from pandas import Series
from shapely.geometry import MultiPoint, Polygon

from syntraj_utileval.datasets.dataset import TrajectoryDataset
from syntraj_utileval.eval import runner
from syntraj_utileval.models import clustering, flow_prediction
from syntraj_utileval.realism import spatial as realism_spatial
from syntraj_utileval.stats import spatio_temporal
from syntraj_utileval.utils.discretize import find_knee, spatial_entropy, step_length


@dataclass
class EvalParams:
    cell_width_m: float
    cell_height_m: float
    global_origin: Tuple[float, float] = (0, 0)
    dbscan_min_samples: int = 5


def eval(method) -> List[float]:
    def wrapper(self, *args, **kwargs):
        results = []
        start_time = time.time()
        for ds in self.synthetics:
            results.append(method(self, ds, *args, **kwargs))
        elapsed = time.time() - start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        if self.verbose:
            method_name = re.sub(r"evaluate_", "", method.__name__)

            print(
                f"**Finished calculating {method_name} in {minutes} minutes {seconds} seconds**"
            )
        return results

    return wrapper


def synthetic_eval_caching(method) -> List[float]:
    def wrapper(self, *args, **kwargs):
        results = []
        start_time = time.time()
        for i, ds in enumerate(self.synthetics):
            if i == 0:
                results.append(method(self, ds, *args, **kwargs, verbose=True))

            else:
                results.append(method(self, ds, *args, **kwargs, verbose=False))

        elapsed = time.time() - start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        method_name = re.sub(r"evaluate_", "", method.__name__)
        if self.verbose:
            print(
                f"**Finished calculating {method_name} in {minutes} minutes {seconds} seconds**"
            )
        return results

    return wrapper


@dataclass
class EvaluationSuite:
    original: TrajectoryDataset
    synthetics: List[TrajectoryDataset]
    params: EvalParams
    data_name: str
    verbose: bool = True

    def __post_init__(self):
        self.original.set_crs()
        self.original.xy_m()

        self.sensitivity_analysis()
        for ds in self.synthetics:
            if ds.metric_crs != self.original.metric_crs:
                raise ValueError(
                    f"CRS mismatch: dataset '{ds.dataset_name}' uses {ds.metric_crs}, "
                    f"but original dataset uses {self.original.metric_crs}. "
                    "All datasets must be reprojected to the same metric CRS "
                    "before discretization."
                )

            # union bbox_m with original dataset
            # use pairwise union bbox in meters of both original and synthetic to ensure full-support calculation
            ds.union_bound_m(self.original)
            ds.discretized(
                cell_w=self.params.cell_width_m,
                cell_h=self.params.cell_height_m,
                global_origin=self.params.global_origin,
            )
        # Fix the crs of the original dataset in params across all synthetic datasets variants
        self.params.metric_crs = self.original.metric_crs
        # assert ds.params.bbox_m == original_bbox_m, "All datasets must share bbox_m"
        self.compute_union_bbox_degree()

    def sensitivity_analysis(self):
        self.original.discretized(
            cell_w=0.5,  # half a meter
            cell_h=0.5,
            global_origin=self.params.global_origin,
        )
        min_x, min_y, max_x, max_y = self.original.bbox_m

        step_meter = step_length(self.original.df_discretized)
        q50 = np.quantile(step_meter, 0.5)
        q75 = np.quantile(step_meter, 0.75)
        q85 = np.quantile(step_meter, 0.85)
        q10 = np.quantile(step_meter, 0.1)

        g_lower = max(
            30, int(q10 // 10) * 10
        )  # ~30 meter, rough estimate of GPS spatial imprecision
        g_upper = int(q50)
        if g_upper < 100:
            g_upper = int(q75)

        unique_tr_fracs, self_tr_fracs, active_cell_fracs = [], [], []
        grid_sizes = np.arange(g_lower, g_upper, 25)
        for grid_size in grid_sizes:
            unique_frac, self_frac, active_frac = spatial_entropy(
                self.original,
                global_origin=self.params.global_origin,
                cell_w=grid_size,
                cell_h=grid_size,
            )
            unique_tr_fracs.append(unique_frac)
            self_tr_fracs.append(self_frac)
            active_cell_fracs.append(active_frac)

        unique_tr_fracs_smooth = (
            Series(unique_tr_fracs).rolling(3, 2, center=True).mean().bfill().ffill()
        )
        self_tr_fracs_smooth = (
            Series(self_tr_fracs).rolling(3, 2, center=True).mean().bfill().ffill()
        )
        active_cell_fracs_smooth = (
            Series(active_cell_fracs).rolling(3, 2, center=True).mean().bfill().ffill()
        )

        g_u = find_knee(grid_sizes, unique_tr_fracs_smooth, "decreasing")
        g_s = find_knee(grid_sizes, self_tr_fracs_smooth, "increasing")
        g_a = find_knee(grid_sizes, active_cell_fracs_smooth, "increasing")
        candidates = np.array([g_u, g_s, g_a])
        candidates = np.array([g for g in [g_u, g_s, g_a] if g is not None])
        if candidates.shape[0] != 0 and self.verbose:
            print("Spatial sensitivity analysis on Original data...")
            print(
                f"Optimal cell edge length is somewhere between: {candidates.min()}m and {candidates.max()}m"
            )

            midpoint = np.median(candidates)
            mid_g = grid_sizes[np.abs(grid_sizes - midpoint).argmin()]
            if (
                np.quantile(step_meter, 0.5) / mid_g > 1
                and np.quantile(step_meter, 0.5) / mid_g < 3
                and self.verbose
            ):
                print(
                    f"Midpoint cel edge: {mid_g}m with a step to grid ratio (SGR) of {np.quantile(step_meter, 0.5) / mid_g} "
                )

    def union_convex_hull_degree(self) -> Polygon:
        all_lats = np.concatenate(
            [self.original.df["lat"].values]
            + [ds.df["lat"].values for ds in self.synthetics]
        )
        all_lons = np.concatenate(
            [self.original.df["lon"].values]
            + [ds.df["lon"].values for ds in self.synthetics]
        )

        all_points = MultiPoint(list(zip(all_lons, all_lats)))

        hull = all_points.convex_hull

        self.union_hull_d = hull
        return self.union_hull_d

    def compute_union_bbox_degree(self) -> Tuple:
        all_lats = np.concatenate(
            [self.original.df["lat"].values]
            + [ds.df["lat"].values for ds in self.synthetics]
        )
        all_lons = np.concatenate(
            [self.original.df["lon"].values]
            + [ds.df["lon"].values for ds in self.synthetics]
        )

        union_bbox_d = (
            all_lons.min(),
            all_lats.min(),
            all_lons.max(),
            all_lats.max(),
        )
        self.union_bbox_d = union_bbox_d
        return self.union_bbox_d

    @eval
    def evaluate_trajectory_lengths(self, ds):
        return runner.trajectory_length_error(self.original, ds)

    @eval
    def evaluate_density(self, ds, partition="all"):
        if partition == "origins":
            return runner.origin_density_error(
                self.original,
                ds,
                x_bins=self.params.cell_width_m,
                y_bins=self.params.cell_height_m,
            )
        if partition == "all":
            return runner.density_error(
                self.original,
                ds,
                x_bins=self.params.cell_width_m,
                y_bins=self.params.cell_height_m,
            )

    @eval
    def evaluate_average_speed(self, ds):
        return runner.average_speed_distribution_similarity(self.original, ds)

    @eval
    def evaluate_I_rank(self, ds):
        return runner.i_rank_similarity(self.original, ds)

    @eval
    def evaluate_traj_clusters_centroid(self, ds):
        return runner.clusters_comparaison(self.original, ds)

    @eval
    def evaluate_pairwise_trajectory_dist(
        self, ds, distance="frechet", normalize_0_1=False, n_jobs=1
    ):
        return runner.pairwise_distance_distribution_similiarity(
            self.original,
            ds,
            distance=distance,
            normalize_0_1=normalize_0_1,
            n_jobs=n_jobs,
        )

    @eval
    def evaluate_reachability(self, ds):
        return spatio_temporal.reachability(ds.df_discretized)

    @eval
    def evaluate_cat_G_rank(self, ds, n_cat: int):
        return runner.G_rank_categorical_similarity(self.original, ds, n_cat)

    @eval
    def evaluate_G_rank(self, ds, n_cat: int):
        return runner.G_rank_similarity(self.original, ds)

    @eval
    def evaluate_transtion_matrix(self, ds, normalize_0_1=False):
        if not hasattr(self, "_original_transition_matrix"):
            P_orig, states_orig, freq_orig = spatio_temporal.transition_matrix(
                self.original.df_discretized
            )
            self._original_transition_matrix = (P_orig, states_orig, freq_orig)

        return runner.transition_similarity_rowwise(
            self.original,
            ds,
            self._original_transition_matrix,
            normalize_0_1=normalize_0_1,
        )

    @eval
    def evaluate_category_loc_top1_match(self, ds):
        return runner.category_loc_top1_match(self.original, ds)

    @eval
    def evaluate_trajectory_plausibility(self, ds, verbose):
        if not hasattr(self, "_implausible_zones_gdf_cache"):
            self._implausible_zones_gdf_cache, self._land_gdf_cache = (
                realism_spatial.get_tags_gdf(
                    bbox=self.union_bbox_d,
                    metric_crs=self.params.metric_crs,
                    verbose=verbose,
                )
            )
        return realism_spatial.trajectory_implausibility(
            ds.df,
            self._implausible_zones_gdf_cache,
            self._land_gdf_cache,
            self.params.metric_crs,
            verbose=verbose,
        )

    @eval
    def evaluate_location_plausibility(self, ds, verbose=True):
        if not hasattr(self, "_implausible_zones_gdf_cache"):
            self._implausible_zones_gdf_cache, self._land_gdf_cache = (
                realism_spatial.get_tags_gdf(
                    bbox=self.union_bbox_d,
                    metric_crs=self.params.metric_crs,
                    verbose=verbose,
                )
            )
        return realism_spatial.location_implausibility(
            ds.df,
            self._implausible_zones_gdf_cache,
            self._land_gdf_cache,
            self.params.metric_crs,
            verbose=verbose,
        )

    @eval
    def evaluate_traj_network_displacement(self, ds, on_road_key, verbose):
        if not hasattr(self, "_network_orig_cache"):
            self._network_orig_cache = realism_spatial.graph_network_infrastructure(
                self.original.df,
                self.original.bbox_d,
                self.original.metric_crs,
                on_road_key=on_road_key,
                verbose=verbose,
            )

        return realism_spatial.trajectory_network_displacement(
            ds.df, self._network_orig_cache
        )

    @eval
    def evaluate_global_flow_prediction(self, ds, normalize_0_1=False):
        if not hasattr(self, "_original_transition_matrix"):
            P_orig, states_orig, freq_orig = spatio_temporal.transition_matrix(
                self.original.df_discretized
            )
            self._original_transition_matrix = (P_orig, states_orig, freq_orig)
        return flow_prediction.global_flow_prediction(
            self.original,
            ds,
            self._original_transition_matrix,
            normalize_0_1=normalize_0_1,
        )

    @eval
    def evaluate_trajectory_next_location_prediction(self, ds, k, normalize_0_1=False):
        if not hasattr(self, "_original_transition_matrix"):
            P_orig, states_orig, freq_orig = spatio_temporal.transition_matrix(
                self.original.df_discretized
            )
            self._original_transition_matrix = (P_orig, states_orig, freq_orig)
        return flow_prediction.trajectory_next_location_prediction(
            self.original,
            ds,
            self._original_transition_matrix,
            normalize_0_1=normalize_0_1,
            k=k,
        )

    @eval
    def evaluate_trajectory_clustering_prediction(
        self,
        ds,
        min_cluster_size=5,
        min_samples=None,
        cluster_selection_epsilon=0.0,  # Optional: set to cell_width to merge small gaps
    ):
        return clustering.trajectory_clustering_prediction(
            self.original,
            ds,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
        )

    def visualize_point_density(self):
        import matplotlib.pyplot as plt

        lat_min, lat_max = self.original.df["lat"].min(), self.original.df["lat"].max()
        lon_min, lon_max = self.original.df["lon"].min(), self.original.df["lon"].max()

        colors = plt.get_cmap("tab10").colors  # tuple of RGBA

        fig, axes = plt.subplots(
            1,
            len(self.synthetics),
            figsize=(5 * len(self.synthetics), 5),
            sharex=True,
            sharey=True,
        )

        if len(self.synthetics) == 1:
            axes = [axes]  # Make axes iterable
        for i, ds in enumerate(self.synthetics):
            axes[i].scatter(
                ds.df["lat"].values,
                ds.df["lon"].values,
                color=colors[i % len(colors)],
                s=1,
                alpha=0.1,
            )
            axes[i].set_title(ds.dataset_name)

        for ax in axes:
            ax.set_xlim(lat_min, lat_max)
            ax.set_ylim(lon_min, lon_max)
            ax.set_aspect("equal", adjustable="box")

        plt.tight_layout()
        plt.show()

    def visualize_map_infrastructure(self):
        import matplotlib.pyplot as plt
        import networkx as nx

        miny, maxy = (
            self.original.df_discretized["y_m"].min(),
            self.original.df_discretized["y_m"].max(),
        )
        minx, maxx = (
            self.original.df_discretized["x_m"].min(),
            self.original.df_discretized["x_m"].max(),
        )

        colors = plt.get_cmap("tab10").colors  # tuple of RGBA
        fig, axes = plt.subplots(
            1,
            len(self.synthetics),
            figsize=(5 * len(self.synthetics), 5),
            sharex=True,
            sharey=True,
        )

        if len(self.synthetics) == 1:
            axes = [axes]  # Make axes iterable

        for i, ds in enumerate(self.synthetics):
            ax = axes[i]

            df_edges = ds.df_discretized[["tid", "cell_idx"]].copy()
            df_edges["prev_cell_idx"] = df_edges.groupby("tid")["cell_idx"].shift(1)
            df_edges.dropna(inplace=True)
            df_edges = df_edges[df_edges["cell_idx"] != df_edges["prev_cell_idx"]]
            edge_counts = (
                df_edges.groupby(["prev_cell_idx", "cell_idx"])
                .size()
                .reset_index(name="weight")
            )

            G = nx.from_pandas_edgelist(
                edge_counts,
                source="prev_cell_idx",
                target="cell_idx",
                edge_attr="weight",
                create_using=nx.DiGraph(),
            )

            cell_centroids = ds.df_discretized.groupby("cell_idx")[
                ["centroid_x", "centroid_y"]
            ].first()
            pos = {
                idx: (row["centroid_x"], row["centroid_y"])
                for idx, row in cell_centroids.iterrows()
            }
            pos = {node: pos[node] for node in G.nodes() if node in pos}

            edges_to_draw = [(u, v) for u, v, w in G.edges(data="weight")]

            nx.draw_networkx_nodes(
                G, pos, ax=ax, node_size=2, node_color="black", alpha=0.1
            )
            nx.draw_networkx_edges(
                G,
                pos,
                ax=ax,
                edgelist=edges_to_draw,
                width=[
                    np.log1p(G.edges[u, v]["weight"]) for u, v in edges_to_draw
                ],  # Thicker lines for more frequent transitions
                edge_color=colors[i],
                alpha=0.6,
                arrowsize=10,
                arrowstyle="->",
            )

            ax.set_title(ds.dataset_name)
            ax.set_xlim(minx, maxx)
            ax.set_ylim(miny, maxy)
            ax.set_aspect("equal", adjustable="box")

        for ax in axes:
            ax.set_xlim(minx, maxx)
            ax.set_ylim(miny, maxy)
            ax.set_aspect("equal", adjustable="box")

        fig.suptitle("Map Infrastructure from Trajectories")
        plt.xlabel("X (meters)")
        plt.ylabel("Y (meters)")
        plt.tight_layout()
        plt.show()

    def visualize_network_infrastructure(self):
        import matplotlib.pyplot as plt
        import networkx as nx

        min_lon, min_lat, max_lon, max_lat = self.union_bbox_d
        north = max_lat
        south = min_lat
        east = max_lon
        west = min_lon
        bbox = (west, south, east, north)

        network_gdf = realism_spatial.osm_tags(
            bbox, self.params.metric_crs, network=True, pois_tags=False, verbose=True
        )

        all_lon = self.original.df["lon"]
        all_lat = self.original.df["lat"]

        reduced_min_lon, reduced_max_lon = np.percentile(all_lon, [5, 95])
        reduced_min_lat, reduced_max_lat = np.percentile(all_lat, [5, 95])
        reduced_north = reduced_max_lat
        reduced_south = reduced_min_lat
        reduced_east = reduced_max_lon
        reduced_west = reduced_min_lon
        reduced_bbox = (reduced_west, reduced_south, reduced_east, reduced_north)

        colors = plt.get_cmap("tab10").colors  # tuple of RGBA
        fig, axes = plt.subplots(
            1,
            len(self.synthetics),
            figsize=(5 * len(self.synthetics), 5),
            sharex=True,
            sharey=True,
        )

        if len(self.synthetics) == 1:
            axes = [axes]  # Make axes iterable

        for i, ds in enumerate(self.synthetics):
            ax = axes[i]
            df_filtered = ds.df[
                (ds.df["lon"] >= west)
                & (ds.df["lon"] <= east)
                & (ds.df["lat"] >= south)
                & (ds.df["lat"] <= north)
            ].copy()
            empirical_network_gdf = realism_spatial.graph_network_infrastructure(
                df_filtered,
                reduced_bbox,
                self.params.metric_crs,
                network_gdf,
                verbose=False,
            )

            flow_counts = empirical_network_gdf["index_right"].value_counts()

            activated_network = network_gdf.loc[flow_counts.index].copy()

            G = nx.DiGraph()

            for idx, row in activated_network.iterrows():
                flow = flow_counts.get(idx, 0)
                coords = list(row.geometry.coords)
                for i in range(len(coords) - 1):
                    u, v = coords[i], coords[i + 1]
                    dist = np.linalg.norm(np.array(u) - np.array(v))
                    G.add_edge(u, v, weight=dist, flow=flow, edge_id=idx)
                    if not row.get("oneway", False):
                        G.add_edge(v, u, weight=dist, flow=flow, edge_id=idx)

            pos = {node: node for node in G.nodes()}
            nx.draw_networkx_nodes(
                G, pos, ax=ax, node_size=2, node_color="black", alpha=0.1
            )

            edges = list(G.edges())
            if not edges:
                ax.set_title(f"{ds.dataset_name} (No data)")
                continue

            weights = [np.log1p(G.edges[e].get("flow", 0)) for e in edges]
            # Normalize weights for better plotting
            max_w = max(weights) if weights else 1
            normalized_widths = [(w / max_w) * 3 for w in weights]

            nx.draw_networkx_edges(
                G,
                pos,
                ax=ax,
                edgelist=edges,
                width=normalized_widths,  # Thicker lines for more frequent transitions
                edge_color=colors[i],
                alpha=0.6,
                arrowsize=10,
                arrowstyle="->",
            )

            ax.set_title(ds.dataset_name)
            ax.set_xlim(min_lon, max_lon)
            ax.set_ylim(min_lat, max_lat)
            ax.set_aspect("equal", adjustable="box")
            break

        for ax in axes:
            ax.set_xlim(reduced_min_lon, reduced_max_lon)
            ax.set_ylim(reduced_min_lat, reduced_max_lat)
            ax.set_aspect("equal", adjustable="box")

        fig.suptitle("Network Infrastructure from Trajectories")
        plt.xlabel("X (degrees)")
        plt.ylabel("Y (degrees)")
        plt.tight_layout()
        plt.show()
