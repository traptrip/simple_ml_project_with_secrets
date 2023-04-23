from pathlib import Path

import timm
import torch
import pytorch_lightning as pl
from pytorch_metric_learning import losses

from dataset import get_dataloaders
from utils import read_config


class Net(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = timm.create_model(
            cfg.net.backbone_name,
            cfg.net.pretrained,
            num_classes=cfg.net.embedding_size,
        )
        self.criterion = losses.ArcFaceLoss(**cfg.loss)
        self.head_lr = cfg.head_lr
        self.backbone_lr = cfg.backbone_lr

    def forward(self, inputs_id, labels=None):
        logits = self.encoder(inputs_id)
        loss = 0
        if labels is not None:
            loss = self.criterion(logits, labels)
        return loss, logits

    def training_step(self, batch, batch_idx):
        data, labels = batch
        loss, logits = self(data, labels)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        data, labels = batch
        loss, logits = self(data, labels)
        self.log("val_loss", loss)

    def cfgure_optimizers(self):
        param_groups = [
            {
                "params": self.encoder.parameters(),
                "lr": self.backbone_lr,
            },
            {
                "params": self.criterion.parameters(),
                "lr": self.head_lr,
            },
        ]
        optimizer = torch.optim.AdamW(param_groups)
        return optimizer


if __name__ == "__main__":
    cfg = read_config(Path(__file__).parent / "../config.yml")

    # prepare data
    train_dl, val_dl = get_dataloaders(Path(cfg.data.data_dir), cfg.train.batch_size)

    # prepare trainer
    net = Net(cfg.train)
    trainer = pl.Trainer(**cfg.train.trainer)

    # train
    trainer.fit(net, train_dl, val_dl)
