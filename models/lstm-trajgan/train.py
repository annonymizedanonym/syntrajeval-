import argparse
import time

import pandas as pd
import tensorflow as tf
from model import LSTM_TrajGAN

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
tf.keras.backend.set_session(session)


import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset", default="weekly_FS-NYC", type=str
    )  # daily_geolife houly_geolife weekly_geolife weekly_FS-NYC
    parser.add_argument("--resume_epoch", default=0, type=int)
    args = parser.parse_args()

    n_epochs = 8000
    n_batch_size = 256
    n_sample_interval = 10

    seed = 2020

    max_length = 144
    latent_dim = 100
    lr = 0.001
    print(f"\nTraining: {args.dataset}\n")

    save_param = f"./training_params/{args.dataset}"

    keys = ["lat_lon", "day", "hour", "category", "mask"]
    vocab_size = {"lat_lon": 2, "day": 7, "hour": 24, "category": 10, "mask": 1}

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
        val = te.copy()
        save_param = f"training_params/{args.dataset}"
        log_name = f"{args.dataset}"

        lat_centroid = (tr["lat"].sum() + val["lat"].sum()) / (len(tr) + len(val))
        lon_centroid = (tr["lon"].sum() + val["lon"].sum()) / (len(tr) + len(val))

        scale_factor = max(
            max(
                abs(tr["lat"].max() - lat_centroid),
                abs(val["lat"].max() - lat_centroid),
                abs(tr["lat"].min() - lat_centroid),
                abs(val["lat"].min() - lat_centroid),
            ),
            max(
                abs(tr["lon"].max() - lon_centroid),
                abs(val["lon"].max() - lon_centroid),
                abs(tr["lon"].min() - lon_centroid),
                abs(val["lon"].min() - lon_centroid),
            ),
        )
    else:
        raise ValueError("The Dataset is undefined")

    train_norm = tr.copy()
    test_norm = te.copy()
    val_norm = val.copy()

    # normalize to deviations

    train_norm["lat"] = tr["lat"].apply(lambda x: x - lat_centroid)
    test_norm["lat"] = te["lat"].apply(lambda x: x - lat_centroid)
    val_norm["lat"] = val["lat"].apply(lambda x: x - lat_centroid)

    train_norm["lon"] = tr["lon"].apply(lambda x: x - lon_centroid)
    test_norm["lon"] = te["lon"].apply(lambda x: x - lon_centroid)
    val_norm["lon"] = val["lon"].apply(lambda x: x - lon_centroid)

    start_load = time.time()

    gan = LSTM_TrajGAN(
        latent_dim,
        keys,
        vocab_size,
        max_length,
        lat_centroid,
        lon_centroid,
        scale_factor,
        learning_rate=lr,
        SEED=seed,
    )

    if args.resume_epoch:
        gan.generator.load_weights(
            f"training_params/{args.dataset}/lr_{lr}_bs_{n_batch_size}_latdim_100/G_model_{str(args.resume_epoch)}.h5"
        )
        gan.discriminator.load_weights(
            f"training_params/{args.dataset}/lr_{lr}_bs_{n_batch_size}_latdim_100/D_model_{str(args.resume_epoch)}.h5"
        )
        gan.combined.load_weights(
            f"training_params/{args.dataset}/lr_{lr}_bs_{n_batch_size}_latdim_100/C_model_{str(args.resume_epoch)}.h5"
        )

    end_load = time.time()

    print(
        f"\n***************** Model loading time: {end_load - start_load:.3f} seconds ******************************************************** \n"
    )

    gan.train(
        train_norm,
        val_norm,
        epochs=n_epochs,
        batch_size=n_batch_size,
        sample_interval=n_sample_interval,
        start_epoch=args.resume_epoch + 1,
        path=save_param,
        tensorboard=True,
        log_name=log_name,
    )
