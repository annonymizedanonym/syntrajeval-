import hashlib
import warnings
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from osmnx._errors import InsufficientResponseError
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)
_RAM_CACHE = {}

TILE_SIZE_DEG = 0.4  # ~45 km at the equator

pois_tags = {
    "boundary": True,
    "landuse": True,
    "place": True,
    "natural": True,
    "waterway": True,
    "building": [
        "yes",
        "house",
        "residential",
        "apartments",
        "commercial",
        "industrial",
        "retail",
        "church",
        "cathedral",
        "garage",
        "parking",
        "bridge",
    ],
    # "healthcare": True,
    "amenity": [
        "restaurant",
        "cafe",
        "fast_food",
        "pub",
        "bar",
        "food_court",
        "biergarten",
        "prison",
        "townhall",
        "police",
        "fire_station",
        "post_office",
        "courthouse",
        "bank",
        "atm",
        "bureau_de_change",
    ],
    # "tourism": True,
    # "leisure": True,
    # "education": True,
    # "shop": True,
    # "office": True,
    # "service": True,
}


def get_tile_indices(bbox, tile_size=TILE_SIZE_DEG):
    """Convert a bbox into absolute tile indices (i,j) based on fixed global origin (-180, -90)."""
    min_lon, min_lat, max_lon, max_lat = bbox
    i_min = int(np.floor((min_lon + 180) / tile_size))
    i_max = int(np.floor((max_lon + 180) / tile_size))
    j_min = int(np.floor((min_lat + 90) / tile_size))
    j_max = int(np.floor((max_lat + 90) / tile_size))
    tiles = [(i, j) for i in range(i_min, i_max + 1) for j in range(j_min, j_max + 1)]
    return tiles


def tile_bbox(i, j, tile_size=TILE_SIZE_DEG):
    """Get bbox of a tile in degrees from its (i,j) index."""
    min_lon = i * tile_size - 180
    max_lon = (i + 1) * tile_size - 180
    min_lat = j * tile_size - 90
    max_lat = (j + 1) * tile_size - 90
    return (min_lon, min_lat, max_lon, max_lat)


def make_cache_key(bbox, metric_crs, layer, pois_tags=None):

    key = {
        "layer": layer,
        "bbox": tuple(round(c, 6) for c in bbox),
        "crs": str(metric_crs),
    }
    if pois_tags is not None:
        key["pois_tags"] = tuple(sorted(pois_tags.items()))
    return hashlib.md5(repr(key).encode()).hexdigest()


def osm_tags(bbox, graph_network, pois_tags, verbose):
    if bool(graph_network) + bool(pois_tags) != 1:
        raise ValueError("Exactly one of `network` or `pois_tags` must be set.")

    tiles = get_tile_indices(bbox)

    if graph_network:
        missing_tiles = []
        all_graphs = []

        for enum, (i, j) in enumerate(tiles):
            NETWORK_DIR = CACHE_DIR / "network"
            NETWORK_DIR.mkdir(parents=True, exist_ok=True)
            tile_key = (f"tile_{i}_{j}", graph_network)
            h = hashlib.md5(str(tile_key).encode()).hexdigest()
            graph_path = NETWORK_DIR / f"{h}.graphml"

            if tile_key in _RAM_CACHE:
                G = _RAM_CACHE[tile_key]
                if verbose and enum == 0:
                    print(
                        f"Loading road graph network for {len(tiles)} tiles from osnx from cache"
                    )
                all_graphs.append(G)
            elif graph_path.exists():
                if verbose and enum == 0:
                    print(
                        f"Loading road graph network for {len(tiles)} tiles from osnx from disk: {graph_path}"
                    )
                G = ox.load_graphml(graph_path)

                all_graphs.append(G)
                _RAM_CACHE[tile_key] = G

            else:
                missing_tiles.append((i, j))

        # Query OSM only for missing tiles      continue
        if len(missing_tiles) > 0:
            if verbose:
                print(
                    "Cache not found for some tiles. Querying OSM API for road graph network. This may take some time ..."
                )

            for i, j in missing_tiles:
                tile_key = (f"tile_{i}_{j}", graph_network)
                h = hashlib.md5(str(tile_key).encode()).hexdigest()
                NETWORK_DIR = CACHE_DIR / "network"
                graph_path = NETWORK_DIR / f"{h}.graphml"
                tile_bbox_coords = tile_bbox(i, j)
                (min_lon, min_lat, max_lon, max_lat) = tile_bbox_coords
                north = max_lat
                south = min_lat
                east = max_lon
                west = min_lon
                tile_bbox_coords = (
                    west,
                    south,
                    east,
                    north,
                )  # osmnx requires this format
                print(f"Querying tile {(i, j)}")
                G = ox.graph_from_bbox(
                    bbox=tile_bbox_coords, network_type=graph_network, simplify=True
                )

                all_graphs.append(G)
                for u, v, k, data in G.edges(keys=True, data=True):
                    data["u"] = u
                    data["v"] = v

                ox.save_graphml(G, graph_path)

                _RAM_CACHE[tile_key] = G

        merged_G = all_graphs[0] if len(all_graphs) == 1 else nx.compose_all(all_graphs)

        return merged_G

    if pois_tags:
        missing_tiles = []
        all_pois_gdf = []

        for i, j in tiles:
            tile_key = (f"tile_{i}_{j}", (tuple(sorted(pois_tags.keys()))))
            h = hashlib.md5(str(tile_key).encode()).hexdigest()
            OSM_TAGS_DIR = CACHE_DIR / "OSM_POI"
            OSM_TAGS_DIR.mkdir(parents=True, exist_ok=True)
            cache_filename = OSM_TAGS_DIR / f"{h}.gpkg"

            if tile_key in _RAM_CACHE:
                if verbose:
                    print(f"Loading PoI tags for tile {(i, j)} from osnx from cache")
                pois_gdf = _RAM_CACHE[tile_key]
                all_pois_gdf.append(pois_gdf)
            elif cache_filename.exists():
                if verbose:
                    print(
                        f"Loading PoI tags for tile {(i, j)} from osnx from disk: {cache_filename}"
                    )
                pois_gdf = gpd.read_file(cache_filename)
                all_pois_gdf.append(pois_gdf)
                _RAM_CACHE[tile_key] = pois_gdf
            else:
                missing_tiles.append((i, j))
                continue

        # Query OSM only for missing tiles
        if len(missing_tiles) > 0:
            OSM_TAGS_DIR = CACHE_DIR / "OSM_POI"
            OSM_TAGS_DIR.mkdir(parents=True, exist_ok=True)
            if verbose:
                print(
                    "Cache not found for some tiles. Querying OSM API for Points of interests. This may take some time ..."
                )
            for i, j in missing_tiles:
                tile_key = (f"tile_{i}_{j}", (tuple(sorted(pois_tags.keys()))))
                h = hashlib.md5(str(tile_key).encode()).hexdigest()

                cache_filename = OSM_TAGS_DIR / f"{h}.gpkg"
                tile_bbox_coords = tile_bbox(i, j)
                (min_lon, min_lat, max_lon, max_lat) = tile_bbox_coords
                north = max_lat
                south = min_lat
                east = max_lon
                west = min_lon
                tile_bbox_coords = (
                    west,
                    south,
                    east,
                    north,
                )  # osmnx requires this format

                all_pois = []
                for idx, (key, v) in enumerate(pois_tags.items()):
                    try:
                        pois = ox.features_from_bbox(tile_bbox_coords, {key: True})
                    except InsufficientResponseError:
                        try:
                            pois = ox.features_from_bbox(tile_bbox_coords, {key: v})
                        except Exception as e:
                            # For other errors (network, etc.), we should still know what happened
                            print("bbox :", tile_bbox_coords)
                            print(f"Unexpected error querying {key}: {e}")
                            pois = gpd.GeoDataFrame(
                                columns=["geometry"], crs="EPSG:4326"
                            )

                    if pois.empty:
                        warnings.warn(
                            f"OSM query returned 0 features for tag {key}={v}."
                            f" You could try specify specific subtags instead"
                        )
                        continue
                    else:
                        # Fix invalid geometries
                        pois["geometry"] = pois["geometry"].buffer(0)
                        pois = pois[pois.is_valid & ~pois.geometry.is_empty]
                        all_pois.append(pois[[key, "geometry"]])

                pois_df = pd.concat(all_pois).drop_duplicates().reset_index(drop=True)

                pois_gdf = gpd.GeoDataFrame(
                    pois_df, geometry="geometry", crs="EPSG:4326"
                )

                pois_gdf.columns = [str(col).lower() for col in pois_gdf.columns]

                pois_gdf = pois_gdf.loc[:, ~pois_gdf.columns.duplicated()]

                pois_gdf.to_file(cache_filename, driver="GPKG")
                all_pois_gdf.append(pois_gdf)

                # merge pis_gdfs
                _RAM_CACHE[tile_key] = pois_gdf
        merged_pois_gdf = gpd.GeoDataFrame(
            pd.concat(all_pois_gdf, ignore_index=True), crs="EPSG:4326"
        )
        merged_pois_gdf = merged_pois_gdf.replace({"none": np.nan, "no": np.nan})
        merged_pois_gdf = merged_pois_gdf.drop_duplicates(subset="geometry")

        return merged_pois_gdf


def trajectory_implausibility(df, implausible_zones_gdf, land_gdf, metric_crs, verbose):
    """
    Calculates the percentage of synthetic centroids that 'hallucinate'
    by appearing in water bodies.

    Parameters
    ----------
    bbox
        Bounding box as `(left, bottom, right, top)`. Coordinates should be in
        unprojected latitude-longitude degrees (EPSG:4326).
    """

    dict_implaus = location_implausibility(
        df, implausible_zones_gdf, land_gdf, metric_crs, verbose
    )
    points_in_water_building_gdf = dict_implaus["labeled_loc_gdf"]
    traj_water_counts = points_in_water_building_gdf.groupby("tid")[
        "is_implausible"
    ].sum()

    trajectories_with_hallucinations = traj_water_counts >= 1

    # import folium

    # anomalous_tids = traj_water_counts[trajectories_with_hallucinations].index.tolist()
    # plot_gdf = points_in_water_building_gdf[
    #     points_in_water_building_gdf["tid"].isin(anomalous_tids)
    # ].copy()
    # plot_gdf = plot_gdf.to_crs("EPSG:4326")
    # m = folium.Map(
    #     location=[40.7128, -74.0060], zoom_start=12, tiles="CartoDB positron"
    # )
    # colors = ["red", "blue", "green", "purple", "orange", "darkred", "cadetblue"]

    # for i, tid in enumerate(anomalous_tids):
    #     traj_data = plot_gdf[plot_gdf["tid"] == tid].sort_index()  # Sort by time/index
    #     color = colors[i % len(colors)]
    #     isolated_points = traj_data[traj_data["is_implausible"]]

    #     # Draw the path (Line)
    #     points = [
    #         [row.geometry.y, row.geometry.x] for _, row in isolated_points.iterrows()
    #     ]
    #     folium.PolyLine(
    #         points, color=color, weight=2.5, opacity=0.8, tooltip=f"TID: {tid}"
    #     ).add_to(m)

    #     # Draw the points (Markers)
    #     for _, row in isolated_points.iterrows():
    #         # Highlight points that are actually "on road"
    #         # fill_color = color if row["is_implausible"] else "white"
    #         folium.CircleMarker(
    #             location=[row.geometry.y, row.geometry.x],
    #             radius=4,
    #             color=color,
    #             fill=True,
    #             fill_color=color,
    #             popup=f"TID: {tid}<br>Dist: {row['dist_to_land']:.2f}m",
    #         ).add_to(m)
    # display(m)
    # m.save("highway_anomalies_map.html")
    return trajectories_with_hallucinations.sum() / df["tid"].nunique()


def get_tags_gdf(metric_crs, bbox, verbose):

    pois_gdf = osm_tags(bbox, graph_network=False, pois_tags=pois_tags, verbose=verbose)

    pois_gdf = pois_gdf.to_crs(metric_crs)

    implausible_zones_gdf = pois_gdf[
        (pois_gdf["natural"].isin(["water", "coastline", "bay", "beach"]))
        | (pois_gdf["place"].isin(["sea", "ocean"]))
        | (pois_gdf["waterway"].notna())
        # Restricted areas
        | (pois_gdf["boundary"].isin(["protected_area"]))
        | (pois_gdf["amenity"].isin(["prison"]))
        | (pois_gdf["landuse"].isin(["military"]))
    ].copy()

    land_gdf = pois_gdf.drop(index=implausible_zones_gdf.index).copy()

    land_gdf = land_gdf[(land_gdf["building"].notna()) | (land_gdf["amenity"].notna())][
        ["building", "amenity", "geometry"]
    ].copy()

    G = osm_tags(bbox, graph_network="drive", pois_tags=False, verbose=verbose)

    G_proj = ox.project_graph(G, to_crs=metric_crs)
    nodes_gdf, network_gdf = ox.graph_to_gdfs(G_proj)

    network_gdf["is_network"] = True

    land_gdf = pd.concat([land_gdf, network_gdf], ignore_index=True)

    return implausible_zones_gdf, land_gdf


def location_implausibility(df, implausible_zones_gdf, land_gdf, metric_crs, verbose):
    """
    Calculates the percentage of synthetic points that 'hallucinate'
    by appearing in implausible areas (water bodies, restricted areas).

    Parameters
    ----------
    bbox
        Bounding box as `(left, bottom, right, top)`. Coordinates should be in
        unprojected latitude-longitude degrees (EPSG:4326).
    """

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["x_m"], df["y_m"]),
        crs=metric_crs,
    )

    points_in_implausible_zones = gpd.sjoin_nearest(
        gdf, implausible_zones_gdf, how="left", distance_col="dist_to_water"
    )

    points_in_implausible_zones = points_in_implausible_zones[
        ~points_in_implausible_zones.index.duplicated(keep="first")
    ]
    if "index_right" in points_in_implausible_zones.columns:
        points_in_implausible_zones = points_in_implausible_zones.drop(
            columns="index_right"
        )

    points_in_implausible_dist_to_land_gdf = gpd.sjoin_nearest(
        points_in_implausible_zones,
        land_gdf,
        how="left",
        distance_col="dist_to_land",
    )
    points_in_implausible_dist_to_land_gdf = points_in_implausible_dist_to_land_gdf[
        ~points_in_implausible_dist_to_land_gdf.index.duplicated(keep="first")
    ]

    points_in_implausible_dist_to_land_gdf.loc[:, "is_implausible"] = (
        points_in_implausible_dist_to_land_gdf["dist_to_water"] < 1e-6
    ) & (points_in_implausible_dist_to_land_gdf["dist_to_land"] > 30)

    return {
        "labeled_loc_gdf": points_in_implausible_dist_to_land_gdf,
        "Implausibility_rate": points_in_implausible_dist_to_land_gdf[
            "is_implausible"
        ].sum()
        / len(gdf),
    }


def graph_network_infrastructure(
    df: pd.DataFrame,
    bbox,
    metric_crs,
    network_gdf=None,
    on_road_key: int = None,
    verbose=True,
):
    df = df.copy()
    """
    Infers a Topological Map from trajectories using a Transition Graph.
    -------
    Returns:
        G: nx.DiGraph with node visit_counts and weighted edges.
    """
    if network_gdf is None:
        G_drive = osm_tags(
            bbox, graph_network="drive", pois_tags=False, verbose=verbose
        )

        G_drive = ox.project_graph(G_drive, to_crs=metric_crs)
        nodes_gdf, edges_gdf = ox.graph_to_gdfs(G_drive)
        edges_gdf = edges_gdf.reset_index()
        network_gdf = edges_gdf

    # Keep only points on the roads if they are identified
    if on_road_key:
        df = df[df["category"] == on_road_key]
        if df.empty:
            raise ValueError(
                "No trajectory points left after filtering for points on the road."
            )
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["x_m"], df["y_m"]),
        crs=metric_crs,
    )

    edge_ids = ox.nearest_edges(G_drive, gdf["x_m"], gdf["y_m"])
    df["u"] = [e[0] for e in edge_ids]
    df["v"] = [e[1] for e in edge_ids]

    # Extract transitions (consecutive transitions)
    df["source"] = df.groupby("tid")["v"].shift()
    df["target"] = df.groupby("tid")["u"].shift(-1)
    transitions = df.dropna(subset=["source", "target"])
    unique_transitions = transitions.drop_duplicates()[["source", "target"]]

    # Precompute full adjacency matrix (CSR), collapsing multi-edges to min weight
    node_list = np.array(G_drive.nodes)
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    edge_weights = {}

    for u, v, d in G_drive.edges(data=True):
        key = (u, v)
        w = d.get("length", 1.0)
        if key not in edge_weights or w < edge_weights[key]:
            edge_weights[key] = w

    rows = [node_to_idx[u] for u, v in edge_weights]
    cols = [node_to_idx[v] for u, v in edge_weights]
    data = list(edge_weights.values())
    A_full = csr_matrix((data, (rows, cols)), shape=(len(node_list), len(node_list)))

    # Batch Dijkstra for all unique sources
    unique_sources = transitions["source"].unique()
    valid_sources = [s for s in unique_sources if s in node_to_idx]
    indices = np.array([node_to_idx[s] for s in valid_sources])
    if verbose:
        print(f"Running batched Dijkstra for {len(indices)} sources...")

    dist_matrix, preds = dijkstra(
        A_full, directed=True, indices=indices, return_predecessors=True
    )

    activated_edge_keys = set()
    jumps = 0
    no_jumps = 0
    from tqdm import tqdm

    source_to_row = {s: i for i, s in enumerate(valid_sources)}
    # def _network_path(transition, G):
    for source, target in tqdm(unique_transitions.itertuples(index=False)):
        if source not in node_to_idx or target not in node_to_idx:
            jumps += 1
            continue

        row_idx = source_to_row[source]
        tgt_idx = node_to_idx[target]

        if np.isinf(dist_matrix[row_idx, tgt_idx]):
            jumps += 1
            continue

        # Backtrack path
        path_indices = []
        curr = tgt_idx
        src_idx = node_to_idx[source]
        while curr != src_idx:
            path_indices.append(curr)
            curr = preds[row_idx, curr]
            if curr == -9999:
                path_indices = []
                break
        if not path_indices:
            # Target is in a different connected component or unreachable due to directed edges.
            jumps += 1
            continue

        path_indices.append(src_idx)
        path_indices = path_indices[::-1]

        for i in range(len(path_indices) - 1):
            u = node_list[path_indices[i]]
            v = node_list[path_indices[i + 1]]
            activated_edge_keys.add((u, v))
            no_jumps += 1
    if verbose:
        print("Total unreachable transitions:", jumps)
        print("Total reachable transitions:", no_jumps)

    # Extract activated edges

    edge_index = network_gdf.set_index(["u", "v"]).index
    mask = edge_index.isin(activated_edge_keys)
    imputed_edges_gdf = network_gdf.loc[mask]

    # empirical_G = nx.Graph()

    # for idx, row in imputed_edges_gdf.iterrows():
    #     empirical_G.add_edge(
    #         row["u"],
    #         row["v"],
    #         edge_id=idx,
    #     )
    # num_components = nx.number_connected_components(empirical_G)

    # print("number of component", num_components)

    return imputed_edges_gdf


def trajectory_network_displacement(
    df: pd.DataFrame, activated_network: gpd.GeoDataFrame
):
    """
    Measures the displacement of trajectories from the
    empirical infrastructure (activated roads).
    """
    # Create high-resolution points along the activated roads
    sampled_pts = []
    for geom in activated_network.geometry:
        # Sample every 10 meters for precision
        for d in np.arange(0, geom.length, 10):
            pt = geom.interpolate(d)
            sampled_pts.append([pt.x, pt.y])

    # Build KD-Tree for O(log n) lookup
    tree = cKDTree(np.array(sampled_pts))

    # Query synthetic coordinates
    syn_coords = df[["x_m", "y_m"]].values
    distances, _ = tree.query(syn_coords)

    # Return Trajectory-Level Metric
    df["displacement"] = distances
    return df.groupby("tid")["displacement"].mean().mean() / 1000  # km
