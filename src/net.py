import timm
import torch
import pytorch_lightning as pl
from pytorch_metric_learning import losses


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
        self.save_hyperparameters()

    def forward(self, inputs, labels=None):
        logits = self.encoder(inputs)
        logits = torch.nn.functional.normalize(logits)
        loss = 0
        if labels is not None:
            loss = self.criterion(logits, labels)
            return loss, logits
        return logits

    def training_step(self, batch, batch_idx):
        data, labels = batch
        loss, logits = self(data, labels)
        self.log("train_loss", loss)
        return {"loss": loss}

    def configure_optimizers(self):
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
