import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from PIL import Image

IMG_EXTS = (".jpg", ".jpeg", ".png")


class Pix2PixTileDataset(Dataset):
    """Returns (mask_onehot 4xHxW, image 3xHxW in [-1,1]).

    Pairs images/masks by stem, accepting .jpg/.jpeg/.png in either folder.
    """

    def __init__(self, root, split="train", tile_size=512, augment=True,
                 tiles_per_image=8, num_classes=4):
        self.img_dir = os.path.join(root, split, "img")
        self.mask_dir = os.path.join(root, split, "mask")
        self.tile_size = tile_size
        self.augment = augment
        self.tiles_per_image = tiles_per_image
        self.num_classes = num_classes

        def _index(dir_path):
            return {
                os.path.splitext(f)[0]: os.path.join(dir_path, f)
                for f in sorted(os.listdir(dir_path))
                if f.lower().endswith(IMG_EXTS)
            }

        imgs = _index(self.img_dir)
        msks = _index(self.mask_dir)
        common = sorted(set(imgs) & set(msks))
        dropped = sorted(set(imgs) - set(msks))
        orphan = sorted(set(msks) - set(imgs))
        if dropped:
            print(f"[Pix2PixTileDataset] WARNING: {len(dropped)} images without "
                  f"mask in {self.img_dir}: {dropped[:5]}")
        if orphan:
            print(f"[Pix2PixTileDataset] WARNING: {len(orphan)} masks without "
                  f"image in {self.mask_dir}: {orphan[:5]}")
        self.pairs = [(imgs[s], msks[s]) for s in common]
        if not self.pairs:
            raise FileNotFoundError(f"no pairs in {root}/{split}")
        self.geo = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(translate_percent=0.1, scale=(0.8, 1.2),
                     rotate=15, p=0.5),
        ])

    def __len__(self):
        if self.augment:
            return len(self.pairs) * self.tiles_per_image
        return len(self.pairs)

    def load_mask(self, path):
        m = np.array(Image.open(path))
        if m.ndim == 3:
            m = m[..., 0]
        return np.clip(np.rint(m.astype(np.float32)),
                       0, self.num_classes - 1).astype(np.uint8)

    def __getitem__(self, idx):
        ip, mp = self.pairs[idx % len(self.pairs)]
        img = np.array(Image.open(ip).convert("RGB"))
        mask = self.load_mask(mp)
        if img.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"Image/mask size mismatch for {ip}: "
                f"{img.shape[:2]} vs {mask.shape[:2]}")
        h, w = mask.shape[:2]
        ts = self.tile_size
        if self.augment:
            max_y, max_x = h - ts, w - ts
            if max_y < 0 or max_x < 0:
                raise ValueError(
                    f"Image {ip} ({h}x{w}) smaller than tile_size {ts}")
            y = random.randint(0, max_y)
            x = random.randint(0, max_x)
            img = img[y:y + ts, x:x + ts]
            mask = mask[y:y + ts, x:x + ts]
            aug = self.geo(image=img, mask=mask)
            img, mask = aug["image"], aug["mask"]
        else:
            img = img[:ts, :ts]
            mask = mask[:ts, :ts]

        onehot = np.zeros((self.num_classes, ts, ts), dtype=np.float32)
        for c in range(self.num_classes):
            onehot[c] = (mask == c).astype(np.float32)
        img_t = torch.from_numpy(img.astype(np.float32).transpose(2, 0, 1)) / 127.5 - 1.0
        return torch.from_numpy(onehot), img_t
