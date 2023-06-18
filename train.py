import logging
from pathlib import Path
from collections import defaultdict

import torch
import pytorch_lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
from tqdm import tqdm

from src.dataset import get_dataloaders
from src.utils import read_config, get_last_logs_dir
from src.net import Net


def init_embeddings_db(train_dl, net, emb_size, batch_size, device, save_path):
    net.to(device)
    net.eval()
    with torch.no_grad():
        embeddings = torch.zeros((len(train_dl.dataset), emb_size))
        labels = []
        for i, (batch, label) in enumerate(tqdm(train_dl)):
            labels.extend(label.tolist())
            embeddings[i : i + batch_size] = net(batch.to(device)).cpu()

        # generate embeddings database for each label
        emb_db = defaultdict(list)
        id2label = train_dl.dataset.id2label
        for i, id_ in enumerate(labels):
            emb_db[id2label[id_]].append(embeddings[i])

        # average embeddings for each label
        for label, embs in emb_db.items():
            emb_db[label] = torch.nn.functional.normalize(
                torch.mean(torch.stack(embs), dim=0, keepdims=True)
            )

    save_path = get_last_logs_dir(Path(save_path)) / "emb_db.pth"
    torch.save(emb_db, save_path)


if __name__ == "__main__":
    cfg = read_config(Path(__file__).parent / "config.yml")

    # prepare data
    logging.info("Prepare dataloaders")
    train_dl, val_dl, _ = get_dataloaders(Path(cfg.data.data_dir), cfg.train.batch_size)

    # prepare trainer
    logging.info("Prepare trainer")
    net = Net(cfg.train)

    logger = TensorBoardLogger(cfg.train.trainer.default_root_dir)
    trainer = pl.Trainer(logger=logger, **cfg.train.trainer)

    # train
    logging.info("Training")
    trainer.fit(net, train_dl, val_dl)

    # prepare embeddings_file
    logging.info("Initialize embeddings DB")
    init_embeddings_db(
        train_dl,
        net,
        cfg.train.net.embedding_size,
        cfg.train.batch_size,
        cfg.train.device,
        cfg.train.save_emb_db_path,
    )
