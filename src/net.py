import timm
import torch
from pytorch_metric_learning import losses


class Net(torch.nn.Module):
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

        self.encoder = self.encoder.to(cfg.device)
        self.criterion = self.criterion.to(cfg.device)

    def forward(self, inputs, labels=None):
        logits = self.encoder(inputs)
        logits = torch.nn.functional.normalize(logits)
        loss = 0
        if labels is not None:
            loss = self.criterion(logits, labels)
            return loss, logits
        return logits

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
