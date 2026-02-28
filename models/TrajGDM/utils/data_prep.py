import numpy as np
import pandas as pd
from pyproj import Transformer


def traj_encoder(trajs, maxx, minx, maxy, miny, cw, ch, num_loc):
    """
    Parameters
    trajs: Any
        array/list of trajectories where each trajectories is of shape (n,2) where shape[1]=3 represernt (x,y) coordinates
    cw: float
        cell width in meters (east-west)
    ch: float
        cell height in meters (north-south)
    """

    coder_trajs = []
    maxi = int(np.ceil((maxx - minx) / cw))
    maxj = int(np.ceil((maxy - miny) / ch))

    def calculate_grid_index(x, y, maxx, minx, maxy, miny, cw, ch):  # , xmin,ymin,xd,yd
        """
        Parameters
        x: float
            coordinates in meters along the x-axis (east-west) (or longitude in degree)
        y: float
            coordinates in meters along the y-axis (north-south) (or latidude in degree)
        """
        if x > maxx or x < minx or y > maxy or y < miny:
            return (-1, -1)
        else:
            i = int((x - minx) // cw)
            j = int((y - miny) // ch)
            return (i, j)

    def raster_coder(i, j, maxi=100, maxj=100):
        """
        Parameters
        i: int
            is the column index (x-direction)
        j: int
            is the row index (y-direction)
        maxi: int
          number of grid cells along the x (east–west) axis (or longitude in degrees)
        maxj: int
            number of grid cells along the y (north–south) axis (or latitude in degrees)
        """
        return int(j * maxi + i)  # row-major indexing (row * width + column)

    for traj in trajs:
        coder_traj = []
        for pt in traj:
            i, j = calculate_grid_index(
                pt[0], pt[1], maxx, minx, maxy, miny, cw, ch
            )  # pt[0]: x-coordinates, pt[0]:y-coordinates
            if i == -1:
                index = num_loc  # 3137  # -1
            else:
                index = raster_coder(i, j, maxi, maxj)
                # Adjust for TrajGDM's internal logic for encoding the actual index (point in the trajectory) and it's neighbors (north/south and east/west)
                # TrajGDM uses 5-point stencil (North, South, East, West, Center) to capture spatial dependencies during the diffusion process
                # By concatenating these 5 embeddings in the spatial_embed layer, the model "learns" not just where a person is, but the "flavor" of the surrounding area.

                locs_safe_input = (
                    index - 1 - maxi
                )  # the TrajGDM model's center is locs_safe +1 + maxi

                # then apply global buffer as TrajGDM (TrajGeneratorNetwork) expects indices to be shifted
                # so that the very first valid cell isn't at index 0 (to avoid padding conflicts).
                # We add a buffer of 'maxi' to keep everything in the positive range.
                index_safe = locs_safe_input + maxi
                if index >= num_loc:
                    print(i, j, index)
            coder_traj.append(index_safe)
        coder_trajs.append(coder_traj)
    return coder_trajs


def get_utm_epsg_from_bbox(bbox):
    min_lon, min_lat, max_lon, max_lat = bbox

    lon_cent = (min_lon + max_lon) / 2.0
    lat_cent = (min_lat + max_lat) / 2.0

    zone = int((lon_cent + 180) // 6) + 1
    zone = max(1, min(zone, 60))

    if lat_cent >= 0:
        epsg_code = f"EPSG:326{zone:02d}"  # Northern hemisphere
    else:
        epsg_code = f"EPSG:327{zone:02d}"  # Southern hemisphere

    return epsg_code


def project_lonlat_to_meter(xd, yd):
    """
    Project geographic coordinates lat/lon (in degrees) to a metric CRS.

    Parameters
    ----------
    xd : np.ndarray
        longitude in degrees.

    yd : np.ndarray
        latitude in degrees.
    metric_crs : str
        Target CRS as EPSG code (e.g., "EPSG:32633").

    Returns
    -------
    xm : np.ndarray
        x/easting (meters)

    ym : np.ndarray
        y/northing (meters)
    """
    xd = np.asarray(xd)
    yd = np.asarray(yd)

    # Infer UTM from mean lon/lat if not provided
    minx = float(np.min(xd))
    miny = float(np.min(yd))
    maxx = float(np.max(xd))
    maxy = float(np.max(yd))
    metric_crs = get_utm_epsg_from_bbox((minx, miny, maxx, maxy))

    transformer = Transformer.from_crs("EPSG:4326", metric_crs, always_xy=True)
    x_m, y_m = transformer.transform(xd, yd)

    return x_m, y_m, metric_crs


def traj_decoder_to_df(encoded_trajs, meta):
    rows = []

    for tid, traj in enumerate(encoded_trajs):
        for idx in traj:
            # Handle Padding (0) or Out-of-Bounds (num_loc)
            if idx == 0 or idx >= meta["num_loc"]:
                continue  # Or handle as None

            # 2. Reverse the +1 shift
            actual_idx = idx - 1

            # 3. Reverse Row-Major Indexing
            # actual_idx = j * maxi + i
            j = actual_idx // meta["maxi"]
            i = actual_idx % meta["maxi"]

            # 4. Convert to Meters (Center of the Cell)
            # We add 0.5 to get the center point of the 500m x 500m square
            x_m = meta["minx"] + (i + 0.5) * meta["cw"]
            y_m = meta["miny"] + (j + 0.5) * meta["ch"]

            rows.append(
                {
                    "tid": tid,  # Trajectory ID
                    "x_m": x_m,
                    "y_m": y_m,
                }
            )

    decoded_df = pd.DataFrame(rows)
    transformer = Transformer.from_crs(meta["crs_metric"], "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(decoded_df["x_m"], decoded_df["y_m"])

    decoded_df["lon"] = lon
    decoded_df["lat"] = lat
    return decoded_df[["tid", "lat", "lon"]]
