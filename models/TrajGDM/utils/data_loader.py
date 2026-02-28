import json
import random

import torch
from torch.utils.data import Dataset  # , Dataloader


class TrajectoryDataset(Dataset):
    def __init__(self, coder_data, max_len, pad_token=0):
        self.num_samples = len(coder_data)
        self.max_len = max_len
        self.pad_token = pad_token
        self.data = torch.full((self.num_samples, max_len), pad_token, dtype=torch.long)
        self.lengths = torch.zeros(self.num_samples, dtype=torch.long)

        for i, traj in enumerate(coder_data):
            trunc_traj = traj[:max_len]
            l = len(trunc_traj)
            self.data[i, :l] = torch.tensor(trunc_traj, dtype=torch.long)
            self.lengths[i] = l

        # Pre-compute the mask once
        self.mask = self.data != pad_token

    def __getitem__(self, index):
        return self.data[index], self.mask[index], self.lengths[index]

    def __len__(self):
        return self.num_samples


def load_trajs(dataset="TDrive", batch_size=8, num_workers=4, train=True, max_len=30):
    if train:
        with open(
            "./datasets/" + dataset + "/" + "Train.json",
            "r",
        ) as file:
            train_trajs = json.loads(file.read())
        try:
            with open(
                "./datasets/" + dataset + "/" + "Val.json",
                "r",
            ) as file:
                val_trajs = json.loads(file.read())
        # for small datasets: weekly, we used the test set as a validation set as well
        except FileNotFoundError:
            with open(
                "./datasets/" + dataset + "/" + "Eval.json",
                "r",
            ) as file:
                val_trajs = json.loads(file.read())

        random.shuffle(train_trajs)
        random.shuffle(val_trajs)

        train_dataset = TrajectoryDataset(coder_data=train_trajs, max_len=max_len)
        val_dataset = TrajectoryDataset(coder_data=val_trajs, max_len=max_len)

        # using collate_fn for fixed lengths samples will only stack them and add the constant lengths
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True,
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=True,
        )

        return train_loader, val_loader

    else:
        with open("./datasets/" + dataset + "/" + "Eval.json", "r") as file:
            test_trajs = json.loads(file.read())

        random.shuffle(test_trajs)
        test_dataset = TrajectoryDataset(coder_data=test_trajs, max_len=max_len)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True,
        )

        return test_loader
