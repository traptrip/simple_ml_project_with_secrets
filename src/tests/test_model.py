from pathlib import Path

import pytest
import torch
import pytorch_lightning as pl

from src.utils import get_last_logs_dir
from src.dataset import get_dataloaders


@pytest.mark.slow
def test_prediction_shape(config, model):
    model_input = torch.zeros(1, 3, 224, 224)
    out = model(model_input)
    assert out.shape == (1, config.train.net.embedding_size)


@pytest.mark.slow
def test_training(config, model):
    train_dl, val_dl, _ = get_dataloaders(
        Path(config.data.data_dir), config.train.batch_size
    )
    trainer = pl.Trainer(**config.train.trainer)
    trainer.fit(model, train_dl, val_dl)

    pl_logs_dir = Path(config.train.trainer.default_root_dir) / "lightning_logs"
    assert pl_logs_dir.exists()

    last_version_path = get_last_logs_dir(pl_logs_dir)
    assert (last_version_path / "checkpoints").exists()
