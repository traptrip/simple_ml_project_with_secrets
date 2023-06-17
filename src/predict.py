from pathlib import Path

import torch
import pandas as pd
from tqdm import tqdm

from utils import read_config
from net import Net
from dataset import get_dataloaders


def predict(net, test_dataloader, emb_db, device, save_path, thresh=0.8):
    train_embs, train_labels = torch.stack(list(emb_db.values())), list(emb_db.keys())
    train_embs = train_embs.squeeze(1)
    labels = [[] for _ in range(len(test_dataloader.dataset))]
    with torch.no_grad():
        # work only with batch_size=1 in dataloader!
        for i, data in enumerate(tqdm(test_dataloader)):
            emb = net(data.to(device))
            cos_sims = torch.nn.functional.cosine_similarity(emb, train_embs, dim=0)
            for j, cs in enumerate(cos_sims):
                if cs > thresh:
                    labels[i].append(train_labels[j])

    # filter empty
    for i, label in enumerate(labels):
        if not label:
            labels[i] = ["new_whale"]

    prediction = pd.DataFrame(
        {
            "Image": [img_p.name for img_p in test_dataloader.dataset.imgs_list],
            "Id": [" ".join(l) for l in labels],
        }
    )
    prediction.to_csv(save_path, index=False)


if __name__ == "__main__":
    cfg = read_config(Path(__file__).parent / "../config.yml")

    _, _, test_dl = get_dataloaders(Path(cfg.infer.dataset_dir), 1)

    ckpt_path = list(Path(cfg.infer.checkpoint_path).glob("*.ckpt"))[0]
    net = Net.load_from_checkpoint(ckpt_path, map_location=cfg.infer.device)
    net.to(cfg.infer.device)
    net.eval()

    emb_db = torch.load(cfg.infer.embeddings_db_path, map_location=cfg.infer.device)

    predict(
        net, test_dl, emb_db, cfg.infer.device, cfg.infer.save_path, cfg.infer.threshold
    )
