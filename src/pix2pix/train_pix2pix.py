import os
import sys
import time
import yaml
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from data import Pix2PixTileDataset
from model import Pix2PixModel

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def load_cfg(path="configs/pix2pix.yaml"):
    with open(resolve(path)) as f:
        return yaml.safe_load(f)


def main(cfg_path="configs/pix2pix.yaml"):
    cfg = load_cfg(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.get("seed", 42))

    ds_root = resolve(cfg["dataset"]["root"])
    train_ds = Pix2PixTileDataset(ds_root, "train",
                                  tile_size=cfg["dataset"]["tile_size"],
                                  augment=True,
                                  tiles_per_image=cfg["dataset"].get("tiles_per_image", 4))
    val_ds = Pix2PixTileDataset(ds_root, "val",
                                tile_size=cfg["dataset"]["tile_size"],
                                augment=False)
    train_dl = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"],
                          shuffle=True, num_workers=4, pin_memory=True,
                          drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=4)

    net = Pix2PixModel(cfg, device)
    t = cfg["train"]
    epochs = t["epochs"]
    ckpt_dir = resolve(cfg.get("ckpt_dir", "checkpoints/pix2pix"))
    sample_dir = os.path.join(PROJECT_ROOT, "runs", "pix2pix_samples")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    step = 0
    for epoch in range(1, epochs + 1):
        net.G.train()
        net.D.train()
        t0 = time.time()
        agg_g = agg_d = 0.0
        for mask, real in train_dl:
            mask, real = mask.to(device), real.to(device)
            loss_g, loss_d = net.train_step(mask, real)
            agg_g += loss_g
            agg_d += loss_d
            step += 1
            if step % t.get("sample_every", 200) == 0:
                net.G.eval()
                with torch.no_grad():
                    fake = net.G(mask[:4])
                grid = torch.cat([real[:4], fake], 0) * 0.5 + 0.5
                save_image(grid, os.path.join(sample_dir, f"step_{step:07d}.png"),
                           nrow=4)
                net.G.train()

        n = len(train_dl)
        print(f"epoch {epoch:03d} | G {agg_g/n:.4f} | D {agg_d/n:.4f} | "
              f"{time.time()-t0:.1f}s")

        if epoch % t.get("ckpt_every", 10) == 0 or epoch == epochs:
            torch.save({"G": net.G.state_dict(), "D": net.D.state_dict(),
                        "epoch": epoch, "cfg": cfg},
                       os.path.join(ckpt_dir, "pix2pix_last.pt"))

        net.G.eval()
        with torch.no_grad():
            for mask, real in val_dl:
                mask, real = mask.to(device), real.to(device)
                fake = net.G(mask)
                grid = torch.cat([real, fake], 0) * 0.5 + 0.5
                save_image(grid, os.path.join(sample_dir, f"val_epoch_{epoch:03d}.png"),
                           nrow=4)
                break

    print("done.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/pix2pix.yaml")
