from pathlib import Path

from omegaconf import OmegaConf
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms


def get_transforms():
    train_transforms = val_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ]
    )
    return train_transforms, val_transforms


def read_config(cfg_path: str):
    return OmegaConf.load(cfg_path)


def get_last_logs_dir(logs_dir: Path):
    versions_paths = [logs_p.name for logs_p in logs_dir.iterdir()]
    return logs_dir / max(versions_paths, key=lambda p: int(p.split("_")[-1]))
