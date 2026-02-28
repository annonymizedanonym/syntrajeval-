import os
import random
import sys

from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), "./TrajGDM"))
sys.path.append(project_root)

import argparse
import gc
import time

import numpy as np
import torch

gc.collect()  # Force Python garbage collection
torch.cuda.empty_cache()  # Clear PyTorch cache
from torch.cuda.amp import GradScaler

# set the seeds
random.seed(2020)
np.random.seed(2020)
torch.manual_seed(2020)

if torch.cuda.is_available():
    torch.cuda.manual_seed(2020)
    torch.cuda.manual_seed_all(2020)  # if you use multi-GPU

torch.set_num_threads(6)  # reduce CPU thread usage
torch.set_num_interop_threads(4)
# torch.cuda.reset_peak_memory_stats()
import json

from models.diffusion import TrajectoryDiffusion
from models.TrajGeneratorNetwork import TrajGeneratorNetwork
from torch.utils.tensorboard import SummaryWriter

# print(project_root)
from utils.data_loader import load_trajs
from utils.metrics import Evaluation_metrics


def trainer(args):
    print(f"training the {args.dataset} dataset")
    device = torch.device("cuda:" + args.cuda if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = load_trajs(
        dataset=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train=True,
        max_len=args.length,
    )

    with open("./datasets/" + args.dataset + "/grid_metadata.json", "r") as f:
        metadata = json.load(f)

    num_loc = metadata["num_loc"]
    maxi = metadata["maxi"]

    LR = 0.0001
    metrics = Evaluation_metrics(num_loc=num_loc, maxi=maxi)

    trajgenerator = TrajGeneratorNetwork(
        num_location=num_loc,
        location_embedding=args.num_hidden,
        maxi=maxi,
        num_head=args.TrajGenerator_heads,
        lstm_hidden=args.num_hidden,
        device=device,
        TrajGenerator_Translayers=args.TrajGenerator_Translayers,
        TrajGenerator_LSTMlayers=args.TrajGenerator_LSTMlayers,
        input_len=args.length,
    ).to(device)

    traj_diffusion = TrajectoryDiffusion(
        model=trajgenerator,
        maxi=maxi,
        lab=2,
        linear_start=0.00085,
        linear_end=0.0120,
        full_n_steps=1000,
    ).to(device)

    optimizer = torch.optim.Adam(
        traj_diffusion.parameters(), lr=LR
    )  # 0.0005 (batch_size 256)

    scaler = GradScaler()

    start_epoch = 0
    best_val_epoch = 0
    best_val_loss = np.inf
    avg_jsd = np.inf
    if args.continue_training_path is not None:
        print(f"continue training the {args.dataset} dataset")
        checkpoint = torch.load(args.continue_training_path, map_location=device)
        traj_diffusion.load_state_dict(checkpoint["model_state_dict"])
        best_val_epoch = checkpoint["epoch"]
        best_val_loss = checkpoint["best_val_loss"]
        best_traj_diffusion = traj_diffusion
        avg_jsd = checkpoint["avg_jsd"]
        start_epoch = best_val_epoch + 1

    model_variant_path = f"l{args.length}_b{args.batch_size}_lr_{LR}"
    checkpoint_path = f"./TrainedModels_{model_variant_path}/{args.dataset}"
    os.makedirs(checkpoint_path, exist_ok=True)

    print("Start Training......")
    writer = SummaryWriter(log_dir=f"runs/{args.dataset}/{model_variant_path}")
    global_step = 0
    for e in range(start_epoch, args.epoch):
        trajgenerator.train()
        traj_diffusion.train()
        train_losses = []
        t1 = time.time()
        for x, mask, lengths in tqdm(
            train_loader,
            desc=f"epoch {e}/{args.epoch}|Lr:{optimizer.param_groups[0]['lr']}|batch",
            bar_format=" {desc}: {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        ):
            x = x.to(device)
            mask = mask.to(device).bool()
            lengths = lengths.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", enabled=False):
                loss = traj_diffusion.generation_training(x, mask, lengths)

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            total_grad_norm = torch.nn.utils.clip_grad_norm_(
                traj_diffusion.parameters(), max_norm=np.inf
            )
            if total_grad_norm > 5.0:
                print(
                    f"Warning: Gradient Norm is exploding: {total_grad_norm:.4f} at step {global_step}"
                )
            writer.add_scalar("Grad/Real_Norm", total_grad_norm, global_step)
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(float(loss.cpu().detach()))
            global_step += 1

        # -- Validation --
        traj_diffusion.eval()
        val_losses = []
        val_trajs_list = []

        with torch.no_grad():
            for x, mask, lengths in tqdm(val_loader, desc=f"Epoch {e} Validation"):
                x, mask, lengths = (
                    x.to(device),
                    mask.to(device).bool(),
                    lengths.to(device),
                )
                with torch.amp.autocast(device_type="cuda", enabled=False):
                    v_loss = traj_diffusion.generation_training(x, mask, lengths)
                val_trajs_list.append(x.cpu())

                val_losses.append(v_loss.item())

        t2 = time.time()

        writer.add_scalar("Loss/epoch_train", np.mean(train_losses), e)
        writer.add_scalar("Loss/epoch_val", np.mean(val_losses), e)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], e)
        print(
            f"training loss {np.mean(train_losses):.5f}| Val Loss {np.mean(val_losses):5f}"
        )

        if np.mean(val_losses) < best_val_loss:
            best_val_epoch = e
            print("saving best model")
            torch.save(
                {
                    "epoch": best_val_epoch,
                    "model_state_dict": traj_diffusion.state_dict(),
                    "best_val_loss": np.mean(val_losses),
                    "avg_jsd": avg_jsd,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "start_LR": LR,
                    "current_LR": optimizer.param_groups[0]["lr"],
                    "hparams": {
                        "num_location": num_loc,
                        "location_embedding": args.num_hidden,
                        "maxi": maxi,
                        "num_head": args.TrajGenerator_heads,
                        "lstm_hidden": args.num_hidden,
                        "TrajGenerator_Translayers": args.TrajGenerator_Translayers,
                        "TrajGenerator_LSTMlayers": args.TrajGenerator_LSTMlayers,
                        "input_len": args.length,
                    },
                },
                checkpoint_path
                + "/trajGDM"
                + "_"
                + str(e)
                + "_"
                + str(float(np.mean(val_losses)))
                + ".pkl",
            )
            best_val_loss = np.mean(val_losses)
            best_traj_diffusion = traj_diffusion
        if e != 0 and e % args.eval_epoch == 0:
            print("Validation: Generating Trajectories......")

            with torch.no_grad():
                # Re-load best weights to ensure we evaluate the best version
                checkpoint = torch.load(
                    f"{checkpoint_path}/trajGDM_{best_val_epoch}_{best_val_loss}.pkl"
                )
                trajgenerator.eval()
                best_traj_diffusion.eval()
                trajs_generated = []
                t1 = time.time()
                for i in range(args.num_evaluation // args.batch_size):
                    trajs = best_traj_diffusion.TrajGenerating(
                        num_samples=args.batch_size, x=None
                    )
                    trajs = torch.argmax(trajs, dim=-1, keepdim=False)
                    trajs_generated.append(trajs)
                    break

                trajs_generated = torch.cat(trajs_generated, dim=0)
                total_val_x = torch.cat(val_trajs_list, dim=0)
                dis_jsd, total_jsd, o_jsd, d_jsd, ratio = metrics.js_traj(
                    real_traj=total_val_x, sythsize_traj=trajs_generated
                )
                t2 = time.time()
                print(
                    "JSD",
                    dis_jsd,
                    total_jsd,
                    o_jsd,
                    d_jsd,
                    ratio,
                    " minutes ",
                    round((t2 - t1) / 60),
                )

                avg_jsd = (dis_jsd + total_jsd + o_jsd + d_jsd) / 4
                writer.add_scalar("metric/validation_gen/total_jsd", total_jsd, e)
                writer.add_scalar("metric/validation_gen/diversity", ratio, e)

                t1 = time.time()
                print("Validation: Reconstructing Trajectories......")
                trajs_reconstructed = []
                for x_val, _, _ in val_loader:
                    reconstracted_val_trajs = traj_diffusion.TrajGenerating(
                        x=x_val.to(device), num_samples=args.batch_size
                    )
                    reconstracted_val_trajs = torch.argmax(
                        reconstracted_val_trajs, dim=-1, keepdim=False
                    )

                    trajs_reconstructed.append(reconstracted_val_trajs)
                    break

                reconstructed = torch.cat(trajs_reconstructed, dim=0)
                dis_jsd, recon_total_jsd, o_jsd, d_jsd, recon_ratio = metrics.js_traj(
                    real_traj=total_val_x, sythsize_traj=reconstructed
                )

                t2 = time.time()

                print(
                    "JSD",
                    dis_jsd,
                    recon_total_jsd,
                    o_jsd,
                    d_jsd,
                    recon_ratio,
                    " minutes ",
                    round((t2 - t1) / 60),
                )
                writer.add_scalar(
                    "metric/validation_recon/total_jsd", recon_total_jsd, e
                )
                writer.add_scalar("metric/validation_recon/diversity", recon_ratio, e)

        else:
            torch.cuda.empty_cache()  # Clear PyTorch cache

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", default=1500, type=int)
    parser.add_argument("--cuda", default="0", type=str)
    parser.add_argument(
        "--dataset", default="hourly_geolife", type=str
    )  # weekly_FS-NYC hourly_geolife geolife_daily
    parser.add_argument("--batch_size", default=64, type=int)  # +128
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--num_hidden", default=512, type=int)
    parser.add_argument("--TrajGenerator_Translayers", default=2, type=int)
    parser.add_argument("--TrajGenerator_heads", default=2, type=int)
    parser.add_argument("--TrajGenerator_LSTMlayers", default=3, type=int)
    parser.add_argument("--num_evaluation", default=256, type=int)
    parser.add_argument("--eval_epoch", default=2000, type=int)
    parser.add_argument(
        "--length", default=144, type=int
    )  # 138 is the max_len but we set it at 130 as we don't have the resources to train with 138 length
    parser.add_argument(
        "--continue_training_path",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--save_path",
        default="./TrainedModels",
        type=str,
    )
    args = parser.parse_args()
    trainer(args)
