import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter

from src.net import Net


class Trainer:
    def __init__(self, config) -> None:
        self.exp_dir = Path(config.exp_dir)
        self.n_epochs = config.n_epochs
        self.device = (
            torch.device(config.device)
            if not isinstance(config.device, torch.device)
            else config.device
        )

        self.net = Net(config)
        self.criterion = self.net.criterion
        self.optimizer = self.net.configure_optimizers()

    def load_ckpt(self, path: Path):
        return torch.load(path, map_location="cpu")

    def train_step(
        self,
        epoch: int,
        data_loader: DataLoader,
    ):
        self.net.train()
        result_loss = 0
        pbar = tqdm(data_loader)
        for bid, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets = targets.to(self.device)
            output = self.net(images)
            loss = self.criterion(output, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            result_loss += loss.item()
            pbar.set_description(
                f"Epoch: {epoch} Train Loss: {(result_loss / (bid + 1)):.6f}"
            )
        result_loss /= self.train_steps
        return result_loss

    def eval_step(
        self,
        epoch: int,
        data_loader: DataLoader,
    ):
        self.net.eval()
        with torch.no_grad():
            result_loss = 0
            pbar = tqdm(data_loader)
            for bid, (images, targets) in enumerate(pbar):
                images = images.to(self.device)
                targets = targets.to(self.device)
                output = self.net(images)
                loss = self.criterion(output, targets)
                result_loss += loss.item()
                pbar.set_description(
                    f"Epoch: {epoch} Eval Loss: {(result_loss / (bid + 1)):.6f}"
                )
            result_loss /= self.eval_steps
        return result_loss

    def train(self, train_loader, eval_loader):
        self.train_steps, self.eval_steps = len(train_loader), len(eval_loader)
        metrics = {"best_loss": float("inf")}
        tb_writer = SummaryWriter(self.exp_dir / "tensorboard_logs")
        logging.info(f"Training for {self.n_epochs} epochs.")
        for epoch in range(self.n_epochs):
            train_loss = self.train_step(epoch, train_loader)
            tb_writer.add_scalar("Train/Loss", train_loss, epoch)
            eval_loss = self.eval_step(epoch, eval_loader)
            tb_writer.add_scalar("Eval/Loss", eval_loss, epoch)
            if metrics["best_loss"] > eval_loss:
                metrics["best_loss"] = eval_loss
                checkpoint = {
                    "epoch": epoch,
                    "metrics": metrics,
                    "net_state": self.net.state_dict(),
                    "criterion_state": self.criterion.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                }
                torch.save(checkpoint, self.exp_dir / "best_checkpoint.pth")

            # save last ckeckpoint
            checkpoint = {
                "epoch": epoch,
                "metrics": metrics,
                "net_state": self.net.state_dict(),
                "criterion_state": self.criterion.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
            }
            torch.save(checkpoint, self.exp_dir / "last_checkpoint.pth")

            epoch += 1

        tb_writer.close()
        logging.info(f"Training was finished")

        best_ckpt = self.load_ckpt(self.exp_dir / "best_checkpoint.pth")
        self.net.load_state_dict(best_ckpt["net_state"])
