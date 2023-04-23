from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from PIL import Image

from utils import get_transforms


def prepare_labels(labels: np.ndarray):
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    return encoded_labels, label_encoder


class CustomDataset(Dataset):
    def __init__(self, dataset_dir, stage="train", df=None, transform=None):
        self.dataset_dir = Path(dataset_dir) / ("train" if stage in ["train", "val"] else "test")
        self.imgs_list = [img for img in self.dataset_dir.iterdir()]
        self.stage = stage
        self.transform = transform
        self.df = df
        if self.stage in ["train", "val"]:
            self.df = self.df.loc[self.df.stage == self.stage].reset_index(drop=True)
            self.labels = df["Id"]

    def __len__(self):
        return len(self.imgs_list)

    def __getitem__(self, idx):
        if self.stage in ["train", "val"]:
            img_name = self.dataset_dir / self.df.loc[idx, "Image"]
            label = self.labels[idx]

        elif self.stage == "test":
            img_name = self.dataset_dir / self.imgs_list[idx]

        img = Image.open(img_name).convert("RGB")
        img = self.transform(img)

        if self.stage in ["train", "val"]:
            return img, label
        elif self.stage == "test":
            return img


def get_dataloaders(dataset_dir: str, batch_size: int):
    train_transform, test_transform = get_transforms()

    metadata_df = pd.read_csv(dataset_dir / "metadata.csv")
    train_ds = CustomDataset(dataset_dir, "train", metadata_df, train_transform)
    val_ds = CustomDataset(dataset_dir, "val", metadata_df, test_transform)

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    return train_dl, val_dl
