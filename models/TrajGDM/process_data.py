import argparse
import json

import numpy as np
import pandas as pd
from utils.data_prep import project_lonlat_to_meter


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
                index = num_loc  # OOB
            else:
                index = raster_coder(i, j, maxi, maxj)

                if index >= num_loc:
                    print(i, j, index)
                if index == 0:
                    print(
                        "index 0, consider shifting by 1 position to avoid confusion with the pad"
                    )

            coder_traj.append(index)
        coder_trajs.append(coder_traj)
    return coder_trajs


def DataSplit_Users(df: pd.DataFrame, frac=2 / 3, SEED=2020):
    frac_parts = []
    rest_parts = []

    for user, user_df in df.groupby("label"):
        user_df = user_df.sort_values("tid")
        unique_tids = user_df["tid"].unique()

        split_point = int(len(unique_tids) * frac)
        user_ids_frac = unique_tids[:split_point]
        user_ids_frac_rest = unique_tids[split_point:]
        frac_parts.append(user_df[user_df["tid"].isin(user_ids_frac)])
        rest_parts.append(user_df[user_df["tid"].isin(user_ids_frac_rest)])

        frac_df = pd.concat(frac_parts).reset_index(drop=True)
        rest_df = pd.concat(rest_parts).reset_index(drop=True)

    return frac_df, rest_df


def df_to_list(df):
    traj_brut = []
    trajs_df = df.groupby("tid")
    for tid, i in trajs_df:
        traj_brut.append(i[["x_m", "y_m"]].values.tolist())
    return traj_brut


def process_data(args):
    df = pd.read_csv(args.lat_lon_path)

    x_m, y_m, crs_metric_utm = project_lonlat_to_meter(
        df["lon"].values, df["lat"].values
    )

    df["x_m"] = x_m
    df["y_m"] = y_m

    x_meter = args.cell_width
    y_meter = args.cell_height

    minx, maxx, miny, maxy = (
        df["x_m"].min(),
        df["x_m"].max(),
        df["y_m"].min(),
        df["y_m"].max(),
    )

    maxi = int(np.ceil((maxx - minx) / x_meter))
    maxj = int(np.ceil((maxy - miny) / y_meter))

    num_loc = int(maxj * maxi) + 1
    print("num_loc", num_loc)
    print("maxi", maxi)

    if args.split_train_frac:
        train, test = DataSplit_Users(df, frac=args.split_train_frac)
        traj_brut_train = df_to_list(train)
        code_train = traj_encoder(
            traj_brut_train, maxx, minx, maxy, miny, x_meter, y_meter, num_loc
        )

        with open(args.save_path + "/" + args.dataset + "/Train.json", "w") as f:
            json.dump(code_train, f)

        if args.split_val_frac:
            test, val = DataSplit_Users(test, frac=args.split_val_frac)

            traj_brut_val = df_to_list(val)
            code_val = traj_encoder(
                traj_brut_val, maxx, minx, maxy, miny, x_meter, y_meter, num_loc
            )
            with open(args.save_path + "/" + args.dataset + "/Val.json", "w") as f:
                json.dump(code_val, f)

            traj_brut_test = df_to_list(test)
            code_test = traj_encoder(
                traj_brut_test, maxx, minx, maxy, miny, x_meter, y_meter, num_loc
            )
            with open(args.save_path + "/" + args.dataset + "/Eval.json", "w") as f:
                json.dump(code_test, f)
        else:
            traj_brut_test = df_to_list(test)
            code_test = traj_encoder(
                traj_brut_test, maxx, minx, maxy, miny, x_meter, y_meter, num_loc
            )
            with open(args.save_path + "/" + args.dataset + "/Eval.json", "w") as f:
                json.dump(code_test, f)

    else:
        traj_brut = []
        trajs_df = df.groupby("tid")
        for tid, i in trajs_df:
            traj_brut.append(i[["x_m", "y_m"]].values.tolist())

        code = traj_encoder(
            traj_brut, maxx, minx, maxy, miny, x_meter, y_meter, num_loc
        )

        with open(args.save_path + "/" + args.dataset + "/Train.json", "w") as f:
            json.dump(code, f)

    grid_metadata = {
        "minx": float(minx),
        "miny": float(miny),
        "maxx": float(maxx),
        "maxy": float(maxy),
        "crs_metric": crs_metric_utm,
        "cw": float(x_meter),
        "ch": float(y_meter),
        "maxi": int(maxi),
        "maxj": int(maxj),
        "num_loc": int(num_loc),
    }

    with open(args.save_path + "/" + args.dataset + "/grid_metadata.json", "w") as f:
        json.dump(grid_metadata, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lat_lon_path",
        default=r"..\Data\weekly_geolife\latlon.csv",
        type=str,
    )
    parser.add_argument(
        "--save_path",
        default="./TrajGDM/datasets",
        type=str,
    )
    parser.add_argument(
        "--dataset", default="weekly_geolife", type=str
    )  # Geolife  TDrive

    parser.add_argument(
        "--cell_width",
        default=500,
        type=int,
    )

    parser.add_argument(
        "--cell_height",
        default=500,
        type=int,
    )

    parser.add_argument(
        "--split_train_frac",
        default=2 / 3,  # 3/4 for hourly_geolife, 2/3 for others
        type=float,
    )

    parser.add_argument(
        "--split_val_frac",
        default=None,  # 1/2 for daily_geolife hourly_geolife
        type=float,
    )

    args = parser.parse_args()
    process_data(args)
