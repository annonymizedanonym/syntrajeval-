import argparse

import pandas as pd


def DataSplit_tid(df: pd.DataFrame, frac=2 / 3, SEED=2020):
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


def split_data(args):
    print(f"********* spliting dataset: {args.dataset} ***********")
    df = pd.read_csv(f"{args.lat_lon_path}\latlon.csv")
    if args.save_path:
        save_path = f"args.save_path/{args.dataset}"
    else:
        save_path = args.lat_lon_path

    train, test = DataSplit_tid(df, frac=args.train_frac)
    train.to_csv(f"{save_path}/train_latlon.csv", index=False)

    if args.val_frac:
        test, val = DataSplit_tid(test, frac=args.val_frac)
        test.to_csv(f"{save_path}/test_latlon.csv", index=False)
        val.to_csv(f"{save_path}/val_latlon.csv", index=False)
    else:
        test.to_csv(f"{save_path}/test_latlon.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lat_lon_path",
        default=r"D:\FluxVision\Data\weekly_geolife",
        type=str,
    )
    parser.add_argument(
        "--save_path",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--dataset", default="weekly_geolife", type=str
    )  # Geolife  TDrive

    parser.add_argument(
        "--train_frac",
        default=2 / 3,  # 3/4 for hourly_geolife 2/3 for others
        type=float,
    )

    parser.add_argument(
        "--val_frac",
        default=None,  # None for weekly geolife  because we don't have enough data
        type=float,
    )

    args = parser.parse_args()
    split_data(args)
