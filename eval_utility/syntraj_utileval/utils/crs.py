import warnings
from typing import Sequence, Tuple

import numpy as np
from pyproj import Transformer


def get_utm_epsg_from_bbox(
    bbox: Tuple[float, float, float, float],
) -> Tuple[str, Tuple[float, float, float, float]]:
    """
    Guess a single UTM EPSG code from lon/lat bounds.
    - UTM is suitable for small-to-moderate areas fully within a single 6° zone and
      latitudes ~[-80°, +84°]. It provides near-uniform metric scaling and low distortion.

    - UTM becomes problematic if:
        * the area spans multiple UTM zones (>~6° longitude width),
        * crosses the antimeridian (±180°),
        * lies outside ~[-80°, +84°] latitude,
      because scale/zone changes introduce distortions or discontinuities.
    - In those cases, a local Azimuthal Equidistant (AEQD) or Lambert Azimuthal Equal Area (LAEA)
      projection centered at the area in question is typically more robust.
    - Although those cases are detected, handling them is not yet implemented.

    Parameters
    ----------
    bbox : Tuple[float, float, float, float]
        Geographic bounds in degrees (EPSG:4326 is the most common), the right order is (left, bottom, right, top) or (min_lon, min_lat, max_lon, max_lat).

    Returns
    -------
    epsg : str
        EPSG code string like "EPSG:32633" (north) or "EPSG:32733" (south).
    bbox
        Bounding box projected to meters in the calculated EPSG code

    Raises
    ------
    ValueError
        If UTM is not appropriate (multi-zone, antimeridian crossing, or latitude out of range).

    Notes
    -----
    - This function intentionally ignores Norway/Svalbard UTM exceptions to stay simple.
    - If you hit a ValueError, choose a local AEQD or LAEA centered at the area’s centroid instead.
    """

    min_lon, min_lat, max_lon, max_lat = bbox

    if min_lat < -80 or max_lat > 84.0:
        raise ValueError(
            "UTM not suitable: latitude out of UTM range (must be within [-80, 84])."
            "Use a local AEQD/LAEA projection"
        )
    if max_lon >= 180 or min_lon < -180:
        warnings.warn(
            "Bounding box is not in [-180, 180), consider normalizing for consistency."
        )

    lon_span = abs(max_lon - min_lon)

    if lon_span > 180.0:
        # Example: [170, -170] -> span is 20° across the 180 meridian; not a single UTM zone
        raise ValueError(
            "UTM not suitable: bounds cross the antimeridian (±180°). "
            "Use a local AEQD/LAEA centered at the area instead."
        )

    # Single-zone check: require footprint to fit well within one 6° zone
    if lon_span > 6.0:
        raise ValueError(
            f"UTM not suitable: longitudinal span is {lon_span:.2f}° (> 6°), likely spans multiple UTM zones."
            "Use a local AEQD/LAEA projection."
        )

    lon_cent = (min_lon + max_lon) / 2.0
    lat_cent = (min_lat + max_lat) / 2.0

    zone = int((lon_cent + 180) // 6) + 1
    zone = max(1, min(zone, 60))

    if lat_cent >= 0:
        epsg_code = f"EPSG:326{zone:02d}"  # Northern hemisphere
    else:
        epsg_code = f"EPSG:327{zone:02d}"  # Southern hemisphere

    x_m, y_m = project_lonlat_to_metric(
        [min_lon, max_lon], [min_lat, max_lat], metric_crs=epsg_code
    )
    return epsg_code, (
        x_m[0],
        y_m[0],
        x_m[1],
        y_m[1],
    )  # same order as min_lon, min_lat, max_lon, max_lat


def project_lonlat_to_metric(
    xd: Sequence[float], yd: Sequence[float], metric_crs: str = None
) -> Tuple[np.ndarray, np.ndarray]:
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
    if metric_crs is None:
        minx = float(np.min(xd))
        miny = float(np.min(yd))
        maxx = float(np.max(xd))
        maxy = float(np.max(yd))
        metric_crs, _ = get_utm_epsg_from_bbox((minx, miny, maxx, maxy))

        warnings.warn(
            f"metric crs was not given, it has been calculated internally based on the input data: {metric_crs}."
        )
        transformer = Transformer.from_crs("EPSG:4326", metric_crs, always_xy=True)
        x_m, y_m = transformer.transform(xd, yd)
    else:
        transformer = Transformer.from_crs("EPSG:4326", metric_crs, always_xy=True)
        x_m, y_m = transformer.transform(xd, yd)

    return x_m, y_m
