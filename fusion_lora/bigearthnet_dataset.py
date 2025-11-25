from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

from typing import Union
class BigEarthNetSpectralDataset(Dataset):
    """
    BigEarthNet-S2 classification dataset for multispectral patches.

    - Loads 6×256×256 tensors from {split}_tensors/
    - Loads multi-label targets from {split}_labels.csv
    """

    def __init__(
        self,
        root: Union[str, Path],
        split: Literal["train", "val", "test"] = "train",
    ):
        super().__init__()
        self.root = Path(root)
        self.split = split

        if split == "train":
            csv_name = "train_labels.csv"
            tensor_dir = "train_tensors"
        elif split == "val":
            csv_name = "validation_labels.csv"
            tensor_dir = "validation_tensors"
        else:
            csv_name = "test_labels.csv"
            tensor_dir = "test_tensors"

        self.tensor_dir = self.root / tensor_dir

        df = pd.read_csv(self.root / csv_name)


        all_paths = [self.tensor_dir / f"{fname}.pt" for fname in df["filename"]]
        exists_mask = [p.exists() for p in all_paths]

        if not all(exists_mask):
            orig_n = len(df)
            df = df[exists_mask].reset_index(drop=True)
            kept_n = len(df)
            print(
                f"[BigEarthNetSpectralDataset] Only {kept_n}/{orig_n} samples have "
                f"tensors in '{tensor_dir}'. Using the existing ones."
            )


        # filename without .pt
        self.filenames = df["filename"].tolist()

        # everything except 'filename' is a class column (0/1)
        self.class_names = list(df.columns[1:])
        labels = df.drop(columns=["filename"]).values.astype("float32")
        self.labels = torch.from_numpy(labels)        # [N, num_classes]
        self.num_classes = self.labels.shape[1]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx: int):
        fname = self.filenames[idx]
        x = torch.load(self.tensor_dir / f"{fname}.pt")   # [6, 256, 256]
        y = self.labels[idx]                              # [num_classes]

        # You can also add height/width if you want
        _, H, W = x.shape
        sample = {
            "image": x,                                   # 6×H×W spectral tensor
            "labels": y,                                  # multi-hot [C]
            "height": H,
            "width": W,
            "meta": {"dataset_name": "bigearthnet_s2"},
        }
        return sample
