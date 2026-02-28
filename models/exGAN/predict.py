import argparse
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from data.csv2npy import data_conversion
from keras.preprocessing.sequence import pad_sequences
from model import LSTM_Traj_exGAN

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
tf.keras.backend.set_session(session)


def Eval_paper(df_original, df_syn):
    from scipy.spatial.distance import directed_hausdorff
    from shapely.geometry import Polygon

    syn_demo_IOU = []
    # Hausdorff
    # syn_best_Haus = []
    syn_demo_Haus = []

    tidlist = list(df_original["tid"].unique())

    for tid in tidlist:
        real_traj = np.concatenate(
            [
                np.array(df_original.loc[df_original["tid"] == tid]["lat"]).reshape(
                    -1, 1
                ),
                np.array(df_original.loc[df_original["tid"] == tid]["lon"]).reshape(
                    -1, 1
                ),
            ],
            axis=1,
        )

        syn_demo_traj = np.concatenate(
            [
                np.array(df_syn.loc[df_syn["tid"] == tid]["lat"]).reshape(-1, 1),
                np.array(df_syn.loc[df_syn["tid"] == tid]["lon"]).reshape(-1, 1),
            ],
            axis=1,
        )

        syn_demo_Haus.append(directed_hausdorff(real_traj, syn_demo_traj)[0])

        real_traj_convex_hull = Polygon(real_traj).convex_hull
        syn_demo_traj_convex_hull = Polygon(syn_demo_traj).convex_hull

        syn_demo_I = real_traj_convex_hull.intersection(syn_demo_traj_convex_hull).area
        syn_demo_U = real_traj_convex_hull.union(syn_demo_traj_convex_hull).area
        syn_demo_IOU.append(syn_demo_I / syn_demo_U)

    return syn_demo_Haus, syn_demo_IOU


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="weekly_geolife", type=str
    )  # daily_geolife weekly_geolife
    parser.add_argument("--sensitive_label", default=4, type=int)

    seed = 2020

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_random_seed(seed)

    args = parser.parse_args()

    n_epochs = 3000

    latent_dim = 100
    max_length = 144

    lr = 0.001
    n_batch_size = 256

    print(
        f"\n********* predicting: {args.dataset}, at epoch {n_epochs} **************************************************** \n"
    )

    param_path = f"training_params/{args.dataset}/G_model_{str(n_epochs)}.h5"
    save_data_path = (
        f"results/syn_traj_test_{args.sensitive_label}_{args.dataset}_{n_epochs}.csv"
    )

    keys = ["lat_lon", "day", "hour", "category", "mask", "except_label"]

    vocab_size = {
        "lat_lon": 2,
        "day": 7,
        "hour": 24,
        "category": 10,
        "mask": 1,
        "except_label": 10,
    }

    if args.dataset == "daily_geolife" or args.dataset == "hourly_geolife":
        tr = pd.read_csv(f"../Data/{args.dataset}/train_latlon.csv")
        te = pd.read_csv(f"../Data/{args.dataset}/test_latlon.csv")
        val = pd.read_csv(f"../Data/{args.dataset}/val_latlon.csv")
        log_name = f"{args.dataset}"

        lat_centroid = (tr["lat"].sum() + te["lat"].sum() + val["lat"].sum()) / (
            len(tr) + len(te) + len(val)
        )
        lon_centroid = (tr["lon"].sum() + te["lon"].sum() + val["lon"].sum()) / (
            len(tr) + len(te) + len(val)
        )

        scale_factor = max(
            max(
                abs(tr["lat"].max() - lat_centroid),
                abs(te["lat"].max() - lat_centroid),
                abs(val["lat"].max() - lat_centroid),
                abs(tr["lat"].min() - lat_centroid),
                abs(te["lat"].min() - lat_centroid),
                abs(val["lat"].min() - lat_centroid),
            ),
            max(
                abs(tr["lon"].max() - lon_centroid),
                abs(te["lon"].max() - lon_centroid),
                abs(val["lon"].max() - lon_centroid),
                abs(tr["lon"].min() - lon_centroid),
                abs(te["lon"].min() - lon_centroid),
                abs(val["lon"].min() - lon_centroid),
            ),
        )

    elif args.dataset == "weekly_geolife" or args.dataset == "weekly_FS-NYC":
        print("test_set is the valset")
        tr = pd.read_csv(f"../Data/{args.dataset}/train_latlon.csv")
        te = pd.read_csv(f"../Data/{args.dataset}/test_latlon.csv")
        save_param = f"training_params/{args.dataset}"
        log_name = f"{args.dataset}"

        lat_centroid = (tr["lat"].sum() + te["lat"].sum()) / (len(tr) + len(te))
        lon_centroid = (tr["lon"].sum() + te["lon"].sum()) / (len(tr) + len(te))

        scale_factor = max(
            max(
                abs(tr["lat"].max() - lat_centroid),
                abs(te["lat"].max() - lat_centroid),
                abs(tr["lat"].min() - lat_centroid),
                abs(te["lat"].min() - lat_centroid),
            ),
            max(
                abs(tr["lon"].max() - lon_centroid),
                abs(te["lon"].max() - lon_centroid),
                abs(tr["lon"].min() - lon_centroid),
                abs(te["lon"].min() - lon_centroid),
            ),
        )
    else:
        raise ValueError("The Dataset is undefined")

    train_norm = tr.copy()
    test_norm = te.copy()

    # normalize to deviations

    test_norm["lat"] = te["lat"].apply(lambda x: x - lat_centroid)
    test_norm["lon"] = te["lon"].apply(lambda x: x - lon_centroid)

    gan = LSTM_Traj_exGAN(
        latent_dim,
        keys,
        vocab_size,
        max_length,
        lat_centroid,
        lon_centroid,
        scale_factor,
        SEED=seed,
    )

    # Test data
    x_test = data_conversion(test_norm, "tid", test=True)

    x_test = [
        x_test[0],
        x_test[1],
        x_test[2],
        x_test[3],
        x_test[4],
        x_test[5].reshape(-1, 1),
        x_test[6].reshape(-1, 1),
    ]
    X_test = [
        pad_sequences(f, max_length, padding="pre", dtype="float64") for f in x_test[:5]
    ]

    num_trajectories = te["tid"].nunique()

    except_label_cond = np.zeros((num_trajectories, gan.vocab_size["category"]))
    except_label_cond[:, args.sensitive_label] = 1
    print(except_label_cond[0])
    except_label_cond = np.repeat(
        except_label_cond[:, np.newaxis, :], max_length, axis=1
    )  # repeat across time steps
    X_test.append(except_label_cond)

    # Add random noise to the data
    noise = np.random.normal(0, 1, (num_trajectories, 100))
    X_test.append(noise)

    # Load params for the generator
    gan.generator.load_weights(param_path)

    # Make predictions
    prediction = gan.generator.predict(X_test)

    traj_attr_concat_list = []
    for attributes in prediction:
        traj_attr_list = []
        idx = 0
        for row in attributes:
            if row.shape == (max_length, 2):
                traj_attr_list.append(row[max_length - x_test[6][idx][0] :])
            else:
                traj_attr_list.append(
                    np.argmax(row[max_length - x_test[6][idx][0] :], axis=1).reshape(
                        x_test[6][idx][0], 1
                    )
                )
            idx += 1
        traj_attr_concat = np.concatenate(traj_attr_list)
        traj_attr_concat_list.append(traj_attr_concat)
    traj_data = np.concatenate(traj_attr_concat_list, axis=1)

    te_encoded = te.copy()
    te_encoded["lat"] = te_encoded["lat"] - gan.lat_centroid
    te_encoded["lon"] = te_encoded["lon"] - gan.lon_centroid

    label = np.array(te_encoded["label"]).reshape(-1, 1)
    tid = np.array(te_encoded["tid"]).reshape(-1, 1)
    traj_data = np.concatenate([label, tid, traj_data], axis=1)
    df_traj_fin = pd.DataFrame(traj_data)

    df_traj_fin.columns = [
        "label",
        "tid",
        "lat",
        "lon",
        "day",
        "hour",
        "category",
        "mask",
        "except",
    ]

    # Convert location deviation to longtitude and latitude
    df_traj_fin["lat"] = df_traj_fin["lat"] + gan.lat_centroid
    df_traj_fin["lon"] = df_traj_fin["lon"] + gan.lon_centroid

    del df_traj_fin["mask"]

    df_traj_fin["tid"] = df_traj_fin["tid"].astype(np.int32)
    df_traj_fin["day"] = df_traj_fin["day"].astype(np.int32)
    df_traj_fin["hour"] = df_traj_fin["hour"].astype(np.int32)
    df_traj_fin["category"] = df_traj_fin["category"].astype(np.int32)
    df_traj_fin["label"] = df_traj_fin["label"].astype(np.int32)

    # Save synthetic trajectory data
    df_traj_fin.to_csv(save_data_path, index=False)

    syn_demo_Haus_infer_paper, syn_demo_IOU_infer_paper = Eval_paper(te, df_traj_fin)

    df = pd.DataFrame(
        {
            # 'syn_best_IOU':syn_best_IOU,
            "syn_demo_IOU": syn_demo_IOU_infer_paper,
            #    'syn_best_Haus':syn_best_Haus,
            "syn_demo_Haus": syn_demo_Haus_infer_paper,
        }
    )
    print(df.describe())
