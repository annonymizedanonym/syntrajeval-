import argparse
import os

import geopandas as gpd
import pandas as pd
from shapely import wkt
from utils.annotation_osm import annotate_geolife, get_osm_tags, reannotate_osm_features
from utils.utils import (
    convert_to_csv,
    extract_stop_segments,
    parallel_fixed_interval_subsample,
    restrict_to_beijing,
    temporal_aggregation,
)


# Geolife dataset URL: https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/
def prep_geolife(args):

    if os.path.exists(args.geolife_csv_path):
        print("Loading the original Geolife in csv format ...")
        df_geolife = pd.read_csv(args.geolife_csv_path)
    else:
        print(
            "Converting Geolife dataset from in plt its original format to csv. This may takes some time..."
        )
        df_geolife = convert_to_csv(
            root_folder=args.geolife_path, save_csv_path=args.geolife_csv_path
        )
    if args.dataset_temporal_aggregation == "weekly":
        threshold = 3600  # seconds
    if args.dataset_temporal_aggregation == "daily":
        threshold = 600
    if args.dataset_temporal_aggregation == "hourly":
        threshold = 25

    df_geolife["datetime"] = pd.to_datetime(df_geolife["datetime"])

    print(
        f"Aggregating the Geolife dataset by {args.dataset_temporal_aggregation} trajectories..."
    )
    temporally_geolife_df, col_tid = temporal_aggregation(
        df_geolife, aggregation_level=args.dataset_temporal_aggregation
    )
    print("Restricting to trajectories that are fully contained in Beijing...")
    beijing_geolife_df = restrict_to_beijing(temporally_geolife_df, col_tid)
    print("Downsampling trajectories...")
    downsampled_geolife_df = parallel_fixed_interval_subsample(
        beijing_geolife_df, col_tid, threshold, n_jobs=-1
    )

    # Filter out trajectories that are too short or too long
    traj_counts = (
        downsampled_geolife_df.groupby(col_tid).size().reset_index(name="points")
    )

    valid_traj_ids = traj_counts[
        (traj_counts["points"] >= 5) & (traj_counts["points"] <= 144)
    ][col_tid]

    downsampled_geolife_filtered_df = downsampled_geolife_df[
        downsampled_geolife_df[col_tid].isin(valid_traj_ids)
    ].copy()
    traj_counts = (
        downsampled_geolife_filtered_df.groupby(col_tid)
        .size()
        .reset_index(name="points")
    )

    print("Detecting stops for annotation...")
    downsampled_geolife_filtered_df_labeled_stop = extract_stop_segments(
        downsampled_geolife_filtered_df, col_tid
    )

    if os.path.exists("./pois_beijing_osm.csv"):
        print("Annotation with OSM database. Hang in there, this takes some time ...")
        pois_by_category = pd.read_csv("./pois_beijing_osm.csv")
        pois_by_category["geometry"] = pois_by_category["geometry"].apply(wkt.loads)
        pois_by_category = gpd.GeoDataFrame(
            pois_by_category, geometry="geometry", crs="EPSG:4326"
        )
    else:
        print("Annotation with OSM database...")
        pois_by_category = get_osm_tags((115.7, 39.4, 117.4, 41.1))

    if os.path.exists("./pois_beijing_osm_recategorized.csv"):
        osm_pois_reannotated = pd.read_csv("./pois_beijing_osm_recategorized.csv")
        osm_pois_reannotated["geometry"] = osm_pois_reannotated["geometry"].apply(
            wkt.loads
        )
        osm_pois_reannotated = gpd.GeoDataFrame(
            osm_pois_reannotated, geometry="geometry", crs="EPSG:4326"
        )
    else:
        osm_pois_reannotated = reannotate_osm_features(pois_by_category)

    geolife_annotated_df = annotate_geolife(
        downsampled_geolife_filtered_df_labeled_stop, osm_pois_reannotated, col_tid
    )

    geolife_annotated_df = geolife_annotated_df.rename(columns={col_tid: "tid"})
    geolife_annotated_df = geolife_annotated_df.rename(columns={"userid": "label"})

    geolife_annotated_df = geolife_annotated_df.rename(columns={"userid": "label"})
    final_geolife = geolife_annotated_df.sort_values(by=["tid", "datetime"])[
        ["label", "tid", "lat", "lon", "day", "hour", "category"]
    ]

    print(f"Saving the {args.dataset_temporal_aggregation} geolife...")
    save_path = f"../{args.dataset_temporal_aggregation}_geolife"
    os.makedirs(save_path, exist_ok=True)
    final_geolife.to_csv(f"{save_path}/latlon.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--geolife_csv_path",
        default="./all_geolife.csv",
        type=str,
    )

    parser.add_argument(
        "--geolife_path",
        default=r"..\Geolife Trajectories 1.3\Data",
        type=str,
    )

    parser.add_argument(
        "--save_path",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--dataset_temporal_aggregation",
        default="weekly",
        type=str,  # daily hourly
    )

    args = parser.parse_args()
    prep_geolife(args)
