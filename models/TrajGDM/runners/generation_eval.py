import gc
import json
import os
import random
import sys
import time

import numpy as np

project_root = os.path.abspath(
    os.path.join(os.path.dirname(os.getcwd()), "./TrajGDM_afterpaper")
)
sys.path.append(project_root)

import argparse

import torch

from models.diffusion import TrajectoryDiffusion
from models.TrajGeneratorNetwork import TrajGeneratorNetwork
from utils.data_loader import load_trajs
from utils.data_prep import traj_decoder_to_df
from utils.metrics import Evaluation_metrics

gc.collect()  # Force Python garbage collection
torch.cuda.empty_cache()  # Clear PyTorch cache
# set the seeds
random.seed(2020)
np.random.seed(2020)
torch.manual_seed(2020)

if torch.cuda.is_available():
    torch.cuda.manual_seed(2020)
    torch.cuda.manual_seed_all(2020)  # if you use multi-GPU

torch.set_num_threads(6)  # reduce CPU thread usage
torch.set_num_interop_threads(4)


def generation_evaluation(args):
    device = torch.device("cuda:" + args.cuda if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        f"./TrainedModels_l144_b64_lr_0.0001/{args.dataset}/trajGDM_{args.epoch}_3.3108416253870185.pkl",
        map_location=device,
    )
    num_loc = checkpoint["hparams"]["num_location"]
    TrajGenerator_heads = checkpoint["hparams"]["num_head"]
    num_hidden = checkpoint["hparams"]["location_embedding"]
    TrajGenerator_Translayers = checkpoint["hparams"]["TrajGenerator_Translayers"]
    TrajGenerator_LSTMlayers = checkpoint["hparams"]["TrajGenerator_LSTMlayers"]
    input_len = checkpoint["hparams"]["input_len"]
    maxi = checkpoint["hparams"]["maxi"]

    trajgenerator = (
        TrajGeneratorNetwork(
            num_location=num_loc,
            location_embedding=num_hidden,
            maxi=maxi,
            num_head=TrajGenerator_heads,
            lstm_hidden=num_hidden,
            device=device,
            TrajGenerator_Translayers=TrajGenerator_Translayers,
            TrajGenerator_LSTMlayers=TrajGenerator_LSTMlayers,
            input_len=input_len,
        )
        .float()
        .to(device)
    )  # double
    traj_diffusion = (
        TrajectoryDiffusion(
            model=trajgenerator,
            maxi=maxi,
            lab=2,
            linear_start=0.00085,
            linear_end=0.0120,
            full_n_steps=1000,
        )
        .float()
        .to(device)
    )

    traj_diffusion.load_state_dict(checkpoint["model_state_dict"])
    traj_diffusion.model_device = device
    traj_diffusion.model.model_device = device
    traj_diffusion.eval()
    dataloader = load_trajs(
        dataset=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train=False,
        max_len=args.length,
    )
    # no need for gradients during inference:
    with torch.no_grad():
        metrics = Evaluation_metrics(num_loc=num_loc, maxi=maxi)

        trajs_list = []
        for x, _, _ in dataloader:
            x.to(device)
            trajs_list.append(x.cpu())
        total_x = torch.cat(trajs_list, dim=0)

        print("Generating Trajectories......")
        if args.according_real:
            trajs_generated = []
            for x, _, _ in dataloader:
                trajs = traj_diffusion.TrajGenerating(
                    x=x.to(device), num_samples=args.batch_size
                )
                trajs = torch.argmax(trajs, dim=-1, keepdim=False)
                trajs_generated.append(trajs)
                generated = torch.cat(trajs_generated, dim=0)
                if args.eval_metrics:
                    dis_jsd, total_jsd, o_jsd, d_jsd, ratio = metrics.js_traj(
                        real_traj=total_x, sythsize_traj=generated
                    )
                    print(
                        "Number of Generated Trajectories:",
                        int(generated.shape[0]),
                        " JSD Moving: ",
                        dis_jsd,
                        " Distribution ",
                        total_jsd,
                        " O-Dis ",
                        o_jsd,
                        " D-Dis ",
                        d_jsd,
                        " Diversity ",
                        ratio,
                    )
        else:
            trajs_generated = []
            for i in range(args.num_evaluation // args.batch_size):
                t1 = time.time()

                trajs = traj_diffusion.TrajGenerating(
                    num_samples=args.batch_size, x=None
                )
                trajs = torch.argmax(trajs, dim=-1, keepdim=False)
                trajs_generated.append(trajs)
                generated = torch.cat(trajs_generated, dim=0)

                t2 = time.time()
                if args.eval_metrics:
                    dis_jsd, total_jsd, o_jsd, d_jsd, ratio = metrics.js_traj(
                        real_traj=total_x, sythsize_traj=generated
                    )
                    print(
                        "Number of Generated Trajectories:",
                        int(generated.shape[0]),
                        " JSD Moving: ",
                        dis_jsd,
                        " Distribution ",
                        total_jsd,
                        " O-Dis ",
                        o_jsd,
                        " D-Dis ",
                        d_jsd,
                        " Diversity ",
                        ratio,
                        "    minutes ",
                        round((t2 - t1) / 60),
                    )
                else:
                    print(
                        "Number of Generated Trajectories:",
                        int(generated.shape[0]),
                        "    minutes ",
                        round((t2 - t1) / 60),
                    )

        if args.save_gen:
            # generate what's left to reach num_eval
            if args.num_evaluation % args.batch_size != 0:
                trajs = traj_diffusion.TrajGenerating(
                    num_samples=args.num_evaluation % args.batch_size, x=None
                )
                trajs = torch.argmax(trajs, dim=-1, keepdim=False)
                trajs_generated.append(trajs)
                generated = torch.cat(trajs_generated, dim=0)
            with open(
                f"./datasets/{args.dataset}/{args.dataset}_GEN_{str(args.length)}.json",
                "w",
            ) as f:
                json.dump(generated.tolist(), f)

            if not args.according_real:
                # convert to latlon in csv
                with open(
                    "./datasets/" + args.dataset + "/grid_metadata.json", "r"
                ) as f:
                    metadata = json.load(f)

                decoded_df = traj_decoder_to_df(
                    generated,
                    metadata,
                )
                decoded_df.to_csv(
                    f"./datasets/{args.dataset}/syn_traj_test_l{str(args.length)}_e{args.epoch}.csv"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default="0", type=str)
    parser.add_argument(
        "--dataset", default="weekly_geolife", type=str
    )  # daily_geolife weeky_geolife
    parser.add_argument("--epoch", default=460, type=int)
    parser.add_argument("--length", default=20, type=int)  # mean length
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--num_evaluation", default=1027, type=int)
    parser.add_argument("--eval_metrics", default=False, type=bool)
    parser.add_argument("--according_real", default=False, type=bool)
    parser.add_argument("--save_gen", default=True, type=bool)
    args = parser.parse_args()
    generation_evaluation(args)
