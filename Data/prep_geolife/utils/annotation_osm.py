from pathlib import Path
from typing import Tuple

import swifter

swifter.set_defaults(progress_bar=False)
import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
import yaml
from shapely.geometry import Point
from tqdm import tqdm

# tqdm.disable()


def get_osm_tags(beijing_bbox: Tuple):
    tags_list = [
        "aeroway",
        "highway",
        "landuse",
        "natural",
        "healthcare",
        "amenity",
        "tourism",
        "leisure",
        "education",
        "shop",
        "office",
        "building",
        "service",
        "public_transport",
        "industrial",
        "railway",
    ]

    all_pois = []

    for tag in tqdm(tags_list):
        pois = ox.features_from_bbox(
            beijing_bbox, {tag: True}
        )  # Download and process POI data from OpenStreetMap for Beijing.
        # Fix invalid geometries
        pois["geometry"] = pois["geometry"].buffer(0)
        pois = pois[pois.is_valid & ~pois.geometry.is_empty]
        all_pois.append(pois)

    pois_by_category = pd.concat(all_pois).drop_duplicates().reset_index(drop=True)
    pois_by_category[tags_list + ["geometry"]].to_csv(
        "./pois_beijing_osm.csv", index=False
    )
    return pois_by_category


def reannotate_osm_features(pois_by_category):
    pois_by_category = pois_by_category.replace({"none": np.nan, "no": np.nan})

    def extract_subcategory(row, column_name, subtags_list):

        matches = row.isin(subtags_list)
        if matches.any():
            # get all matching value (from all columns)
            value = row[matches].tolist()
            # set the matching cell(s) to NaN
            row[matches] = np.nan

            # put the found value into the target column in the form of string to avoidambiguous values in join
            row[column_name] = ",".join(map(str, value))
        return row

    path = Path("./config.yml")
    with path.open("r", encoding="utf-8") as f:
        pois_categories = yaml.safe_load(f)

    all_categories = []
    for key, values in pois_categories.items():
        all_categories.append(key)
        pois_by_category = pois_by_category.swifter.apply(
            extract_subcategory,
            axis=1,
            column_name=key,
            subtags_list=values,  # values is a list
        )

    # pois_by_category = pois_by_category.drop("service", axis=1)
    # pois_by_category = pois_by_category.drop("tourism", axis=1)
    # pois_by_category = pois_by_category.drop("industrial", axis=1)
    # pois_by_category = pois_by_category.drop("railway", axis=1)
    # pois_by_category = pois_by_category.drop("landuse", axis=1)
    # pois_by_category = pois_by_category.drop("office", axis=1)
    # pois_by_category = pois_by_category.drop("shop", axis=1)
    # pois_by_category = pois_by_category.drop("building", axis=1)
    # pois_by_category = pois_by_category.drop("amenity", axis=1)
    # pois_by_category = pois_by_category.drop("public_transport", axis=1)
    pois_by_category[all_categories + ["geometry"]].to_csv(
        "./pois_beijing_osm_recategorized.csv", index=False
    )
    return pois_by_category[all_categories + ["geometry"]]


def annotate_geolife(df_stop_labeled, osm_pois_reannotated, col_tid):
    unique_staypoints = (
        df_stop_labeled[df_stop_labeled["stop"]][["lat", "lon"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    unique_staypoints["geometry"] = unique_staypoints.apply(
        lambda row: Point(row["lon"], row["lat"]), axis=1
    )
    staypoints_gdf_unbuffered = gpd.GeoDataFrame(
        unique_staypoints, geometry="geometry", crs="EPSG:4326"
    )  # originally longitude/latitude (EPSG:4326)

    staypoints_gdf_unbuffered.plot(color="blue", markersize=10)

    # Create geometry column for the dataframe where staypoints are identified (True) and points of movement are identified (False)
    df_stop_labeled["geometry"] = df_stop_labeled.apply(
        lambda row: Point(row["lon"], row["lat"]), axis=1
    )
    gdf_annotated = gpd.GeoDataFrame(
        df_stop_labeled, geometry="geometry", crs="EPSG:4326"
    )
    gdf_annotated[["userid", col_tid, "lat", "lon", "stop", "geometry"]].sort_values(
        col_tid
    )

    staypoints_gdf_buffered = staypoints_gdf_unbuffered.to_crs("EPSG:3857")
    osm_pois_reannotated = osm_pois_reannotated.to_crs("EPSG:3857")

    staypoints_gdf_buffered["staypoint_id"] = staypoints_gdf_buffered.index
    staypoints_gdf_buffered["original_geometry"] = staypoints_gdf_buffered["geometry"]
    staypoints_gdf_buffered["geometry"] = staypoints_gdf_buffered.geometry.buffer(50)

    joined = gpd.sjoin(
        staypoints_gdf_buffered,
        osm_pois_reannotated,
        how="left",
        predicate="intersects",
    )

    mapping_counts = joined["staypoint_id"].value_counts().sort_index()

    categories = [c for c in osm_pois_reannotated.columns if c != "geometry"]
    category_map = {col: i + 1 for i, col in enumerate(categories)}

    def assign_category(row):
        for col in list(category_map):
            if pd.notnull(row[col]):
                return category_map[col]
        return np.nan  # No category found

    joined["category"] = joined.apply(assign_category, axis=1)
    # Counting and Selecting Most Frequent Category
    category_counts = (
        joined.groupby(["staypoint_id", "category"]).size().reset_index(name="count")
    )

    # Keep only the most frequent category per staypoint
    most_frequent_counts = category_counts.drop_duplicates(
        subset=["staypoint_id"], keep="first"
    )
    most_frequent_join = staypoints_gdf_buffered.merge(
        most_frequent_counts[["staypoint_id", "category"]],
        on="staypoint_id",
        how="left",
    )
    (
        len(most_frequent_join),
        len(staypoints_gdf_buffered),
        most_frequent_join.groupby(["category"]).size(),
    )

    # Staypoints left without categories are assigned to a new categories called other
    num_categories = most_frequent_join["category"].nunique()
    print("num categ", num_categories)
    most_frequent_join["category"] = (
        most_frequent_join["category"].fillna(len(categories) + 1).astype(int)
    )
    most_frequent_join.groupby(["category"]).size()

    geo_unique_poi_category = most_frequent_join.copy()
    geo_unique_poi_category = geo_unique_poi_category.set_geometry("original_geometry")

    gdf_annotated = gdf_annotated.reset_index(
        drop=True
    )  # ensure a unique identifier for each row # reindex to ensure all original points are present
    gdf_annotated["orig_idx"] = gdf_annotated.index

    joined_total = gpd.sjoin(
        gdf_annotated.to_crs("EPSG:3857"),
        geo_unique_poi_category[["original_geometry", "category"]],
        how="left",
        predicate="within",
    )  # within because i need exact mtaches of points

    # assign categories with NaN (points when the user is on the move)
    joined_total["category"] = joined_total["category"].fillna(0).astype(int)

    cat_label_final = {v: k for k, v in category_map.items()}
    cat_label_final[0] = "on the road"  # refering to driving on the road
    cat_label_final[len(category_map) + 1] = "other"
    joined_total["category_label"] = joined_total["category"].map(cat_label_final)
    return joined_total
