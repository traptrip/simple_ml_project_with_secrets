from pathlib import Path

import pytest
import torch

from src.dataset import get_dataloaders
from src.training import Trainer


@pytest.mark.slow
def test_prediction_shape(config, model):
    model_input = torch.zeros(1, 3, 224, 224)
    out = model(model_input)
    assert out.shape == (1, config.train.net.embedding_size)


@pytest.mark.slow
def test_training(config):
    train_dl, val_dl, _ = get_dataloaders(
        Path(config.data.data_dir), config.train.batch_size
    )
    trainer = Trainer(config.train)
    trainer.train(train_dl, val_dl)

    logs_dir = Path(config.train.exp_dir)
    assert logs_dir.exists()
    assert (logs_dir / "best.ckpt").exists()
