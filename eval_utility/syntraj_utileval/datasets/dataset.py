import warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from syntraj_utileval.utils import crs


@dataclass
class TrajectoryDataset:
    df: pd.DataFrame
    dataset_name: str
    lon_col: str
    lat_col: str
    trajectoryid_col: str
    userid_col: str = None
    datetime_col: str = None
    semantic_category_col: str = None
    _x_m: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _y_m: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.df = self.df.copy()
        self.df.rename(columns={self.lon_col: "lon"}, inplace=True)
        self.df.rename(columns={self.lat_col: "lat"}, inplace=True)
        self.df.rename(columns={self.trajectoryid_col: "tid"}, inplace=True)
        if self.semantic_category_col:
            if self.semantic_category_col in self.df.columns:
                self.df.rename(
                    columns={self.semantic_category_col: "category"}, inplace=True
                )
            else:
                warnings.warn(
                    f"Column: '{self.semantic_category_col}' does not exist for dataset '{self.dataset_name}'."
                )
        if self.datetime_col:
            if self.datetime_col in self.df.columns:
                self.df.rename(columns={self.datetime_col: "datetime"}, inplace=True)
                self.df["datetime"] = pd.to_datetime(self.df["datetime"])
            else:
                warnings.warn(
                    f"Column: '{self.datetime_col}' does not exist for dataset '{self.dataset_name}'."
                )

        # Sort the coordinates
        try:
            self.df = self.df.sort_values(["tid", "datetime"])

        except KeyError:
            self.df = self.df.reset_index().sort_values(["tid", "index"])
        self.set_crs()
        self.xy_m()
        self.compute_bbox_d()
        self.compute_bbox_m()

    def set_crs(self):
        min_lon = float(np.min(self.df["lon"].values))
        min_lat = float(np.min(self.df["lat"].values))
        max_lon = float(np.max(self.df["lon"].values))
        max_lat = float(np.max(self.df["lat"].values))
        metric_crs, bbox_m = crs.get_utm_epsg_from_bbox(
            (min_lon, min_lat, max_lon, max_lat)
        )
        self.metric_crs = metric_crs
        self.bbox_m = bbox_m
        return self.metric_crs, self.bbox_m

    def xy_m(self) -> tuple[np.ndarray, np.ndarray]:
        if self._x_m is None or self._y_m is None:
            x_m, y_m = crs.project_lonlat_to_metric(
                self.df["lon"].values,
                self.df["lat"].values,
                self.metric_crs,
            )

            self.df["x_m"] = x_m
            self.df["y_m"] = y_m
            self._x_m = x_m
            self._y_m = y_m

        return self._x_m, self._y_m

    def discretized(
        self,
        cell_w: Optional[float],
        cell_h: Optional[float],
        global_origin: Tuple[float, float],
    ) -> pd.DataFrame:
        cw = float(cell_w)
        ch = float(cell_h)
        x_m, y_m = self._x_m, self._y_m

        ox, oy = global_origin
        j = np.floor((x_m - ox) / cw).astype(np.int64)
        i = np.floor((y_m - oy) / ch).astype(np.int64)

        df_discretized = self.df.copy()
        df_discretized["cell_j"] = j
        df_discretized["cell_i"] = i
        df_discretized["cell_idx"] = (i.astype(np.int64) << 32) | (
            j.astype(np.int64) & 0xFFFFFFFF
        )

        df_discretized["centroid_x"] = ox + (df_discretized["cell_j"] + 0.5) * cw
        df_discretized["centroid_y"] = oy + (df_discretized["cell_i"] + 0.5) * ch

        self.df_discretized = df_discretized

        # Add centroids of cells
        unique_cells = df_discretized[
            ["cell_idx", "cell_i", "cell_j"]
        ].drop_duplicates()
        cell_centroids = {
            row.cell_idx: (
                (row.cell_j + 0.5) * cw,
                (row.cell_i + 0.5) * ch,
            )
            for row in unique_cells.itertuples(index=False)
        }

        self.cell_centroids = cell_centroids
        # self.nx = nx
        # self.ny = ny
        return self.df_discretized

    def compute_bbox_m(self):
        """
        compute bounding boxes in meter
        """

        bbox_m = (
            self._x_m.min(),
            self._y_m.min(),
            self._x_m.max(),
            self._y_m.max(),
        )

        self.bbox_m = bbox_m
        return self.bbox_m

    def compute_bbox_d(self):
        """
        compute union bounding boxes in degree of the synthetic dataset and the original one
        """

        bbox_d = (
            self.df["lon"].min(),
            self.df["lat"].min(),
            self.df["lon"].max(),
            self.df["lat"].max(),
        )
        self.bbox_d = bbox_d
        return self.bbox_d

    def union_bound_m(self, original):
        """
        compute union bounding boxes in meter of the synthetic dataset and the original one
        """
        all_x = np.concatenate([original._x_m, self._x_m])
        all_y = np.concatenate([original._y_m, self._y_m])
        union_bbox_m = (
            all_x.min(),
            all_y.min(),
            all_x.max(),
            all_y.max(),
        )
        self.union_bbox_m = union_bbox_m
        return self.union_bbox_m

    def union_bound_d(self, original):
        """
        compute union bounding boxes in degree of the synthetic dataset and the original one
        """
        all_lon = np.concatenate([original.df.lon, self.df.lon])
        all_lat = np.concatenate([original.df.lat, self.df.lat])
        union_bbox_d = (
            all_lon.min(),
            all_lat.min(),
            all_lon.max(),
            all_lat.max(),
        )
        self.union_bbox_d = union_bbox_d
        return self.union_bbox_d
