import os
from pathlib import Path

import numpy as np
import pandas as pd
import swifter
from geopy.distance import great_circle
from joblib import Parallel, delayed
from tqdm import tqdm

swifter.set_defaults(progress_bar=False)
# tqdm.disable()


def temporal_aggregation(df, aggregation_level):
    """
    Create a temporal aggregation: week, day, hour
    """
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.dayofweek
    df["year"] = df["datetime"].dt.isocalendar().year
    df["week"] = df["datetime"].dt.isocalendar().week
    col_tid = f"{aggregation_level}_tid"
    if aggregation_level == "weekly":
        df[col_tid] = df.groupby(["userid", "year", "week"]).ngroup()
        temporally_df = df[
            [
                "userid",
                "weekly_tid",
                "lat",
                "lon",
                "week",
                "year",
                "hour",
                "day",
                "datetime",
            ]
        ].copy()
        return temporally_df, col_tid

    if aggregation_level == "daily":
        df["daily_tid"] = df.groupby(["userid", "year", "week", "day"]).ngroup()
        temporally_df = df[
            [
                "userid",
                "daily_tid",
                "lat",
                "lon",
                "week",
                "year",
                "hour",
                "day",
                "datetime",
            ]
        ].copy()
        return temporally_df, col_tid
    if aggregation_level == "hourly":
        df["hourly_tid"] = df.groupby(
            ["userid", "year", "week", "day", "hour"]
        ).ngroup()
        temporally_df = df[
            [
                "userid",
                "hourly_tid",
                "lat",
                "lon",
                "week",
                "year",
                "hour",
                "day",
                "datetime",
            ]
        ].copy()
        return temporally_df, col_tid


def convert_to_csv(root_folder, save_csv_path=None):
    """
    Convert the original geolife file to a DataFrame with the follwing columns:
    userid, tid, lat,lon, date, time
    """
    current_tid = 0
    rows = []
    for folder_name in tqdm(os.listdir(root_folder)):
        folder_path = os.path.join(root_folder, folder_name, "Trajectory")
        for plt_file in Path(folder_path).glob("*.plt"):
            with plt_file.open("r") as file:
                traj = file.readlines()[
                    6:
                ]  # each plt file for each user represents a trajectory
            for point in traj:
                point = point.strip().split(",")
                lat = float(point[0])
                lon = float(point[1])
                altitude = float(point[3])
                ts_date = point[5]
                ts_time = point[6]
                rows.append(
                    {
                        "userid": int(folder_name),
                        "tid": current_tid,
                        "lat": lat,
                        "lon": lon,
                        "altitude": altitude,
                        "date": ts_date,
                        "time": ts_time,
                        "plt_name": plt_file.name,
                    }
                )
            current_tid += 1

    df = pd.DataFrame(
        rows,
        columns=["userid", "tid", "lat", "lon", "altitude", "date", "time", "plt_name"],
    )
    df["datetime"] = df["date"] + " " + df["time"]
    df = df.drop(columns=["date", "time", "altitude", "plt_name"])
    df.to_csv(save_csv_path, index=False)
    return df


def restrict_to_beijing(df, col_tid):

    df["inside_beijing"] = (
        (df["lat"] > 39.4)
        & (df["lat"] < 41.1)
        & (df["lon"] > 115.4)
        & (df["lon"] < 117.6)
    )

    tid_fully_inside_beijing = df.groupby(col_tid)[
        "inside_beijing"
    ].all()  # True =  Trajectory has all points inside of beijing

    geo_daily_beijing_fully_inside = df[
        df[col_tid].isin(tid_fully_inside_beijing[tid_fully_inside_beijing].index)
    ].copy()

    return geo_daily_beijing_fully_inside


def parallel_fixed_interval_subsample(df, col_tid, threshold, n_jobs=-1):

    def fixed_interval_subsample(df, interval_seconds):
        df = df.sort_values("datetime").reset_index(drop=True)
        start_time = df["datetime"].iloc[0]
        df["interval"] = (
            (df["datetime"] - start_time).dt.total_seconds() // interval_seconds
        ).astype(int)
        selected_points = []
        for _, group in df.groupby("interval"):
            mean_lat = group["lat"].mean()
            mean_lon = group["lon"].mean()
            dists = np.sqrt(
                (group["lat"] - mean_lat) ** 2 + (group["lon"] - mean_lon) ** 2
            )
            closest_idx = dists.idxmin()
            selected_points.append(group.loc[closest_idx])
        return pd.DataFrame(selected_points).reset_index(drop=True)

    grouped = [group for _, group in df.groupby(col_tid)]
    results = Parallel(n_jobs=n_jobs)(
        delayed(fixed_interval_subsample)(group, threshold) for group in grouped
    )
    return pd.concat(results)


def extract_stop_segments(df, col_tid):

    def _compute_speed(df):
        df = df.sort_values("datetime").copy()
        df["lat_shift"] = df["lat"].shift(-1)
        df["lon_shift"] = df["lon"].shift(-1)
        df["time_shift"] = df["datetime"].shift(-1)

        # Calculate distance in meters
        df["dist_m"] = df.apply(
            lambda row: (
                great_circle(
                    (row["lat"], row["lon"]), (row["lat_shift"], row["lon_shift"])
                ).meters
                if pd.notnull(row["lat_shift"])
                else np.nan
            ),
            axis=1,
        )

        # Calculate time difference in seconds
        df["dt_s"] = (df["time_shift"] - df["datetime"]).dt.total_seconds()

        # Speed in km/h
        df["speed_kmh"] = (df["dist_m"] / df["dt_s"]) * 3.6
        return df

    def _assign_mode(speed):
        if pd.isnull(speed):
            return np.nan
        elif speed < 3:
            return "static"
        elif speed < 9:
            return "pedestrian"
        elif speed < 40:
            return "driving in the city-cycling"
        else:
            return "Rural drives/highways"

    df = df.groupby(col_tid, group_keys=False).apply(_compute_speed)
    df = df.reset_index(drop=True)

    df["mode"] = df["speed_kmh"].swifter.apply(_assign_mode)
    print(
        "nb roads",
        len(
            df[
                (
                    (
                        df["mode"].isin(
                            ["driving in the city-cycling", "Rural drives/highways"]
                        )
                    )
                    & (df["mode"].isna())
                )
            ],
        ),
    )

    def extract_stop_segments_traj(traj, col_tid):

        traj = traj.sort_values("datetime").reset_index(drop=True)
        traj["stop"] = False  # Default: not a stop
        traj = traj.copy()
        traj["stop"] = traj["mode"].isin(["pedestrian", "static"])

        # Mark first and last point of the trajectory as stop
        traj.iloc[0, traj.columns.get_loc("stop")] = True  # first point
        traj.iloc[-1, traj.columns.get_loc("stop")] = True  # last point

        return traj

    # parallelize
    groups = [group for _, group in df.groupby(col_tid)]
    results = Parallel(n_jobs=-1)(
        delayed(extract_stop_segments_traj)(group, col_tid) for group in groups
    )
    df_stop_labeled = pd.concat(results, ignore_index=True)
    return df_stop_labeled
