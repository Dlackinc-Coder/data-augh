import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image


class Pix2PixTileDataset(Dataset):
    """Returns (mask_onehot 4xHxW, image 3xHxW in [-1,1])."""

    def __init__(self, root, split="train", tile_size=512, augment=True,
                 tiles_per_image=8):
        self.img_dir = os.path.join(root, split, "img")
        self.mask_dir = os.path.join(root, split, "mask")
        self.tile_size = tile_size
        self.augment = augment
        self.tiles_per_image = tiles_per_image
        self.stems = sorted(
            f[:-len(os.path.splitext(f)[1])]
            for f in os.listdir(self.img_dir) if f.lower().endswith(".jpg"))
        self.pairs = []
        for s in self.stems:
            ip = os.path.join(self.img_dir, s + ".jpg")
            mp = os.path.join(self.mask_dir, s + ".jpg")
            if os.path.exists(mp):
                self.pairs.append((ip, mp))
        if not self.pairs:
            raise FileNotFoundError(f"no pairs in {root}/{split}")
        self.geo = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(translate_percent=0.1, scale=(0.8, 1.2),
                     rotate=15, cval=0, p=0.5),
        ])

    def __len__(self):
        if self.augment:
            return len(self.pairs) * self.tiles_per_image
        return len(self.pairs)

    @staticmethod
    def load_mask(path):
        m = np.array(Image.open(path))
        if m.ndim == 3:
            m = m[..., 0]
        return np.clip(np.rint(m.astype(np.float32)), 0, 3).astype(np.uint8)

    def __getitem__(self, idx):
        ip, mp = self.pairs[idx % len(self.pairs)]
        img = np.array(Image.open(ip).convert("RGB"))
        mask = self.load_mask(mp)
        h, w = mask.shape
        ts = self.tile_size
        if self.augment:
            y = random.randint(0, h - ts)
            x = random.randint(0, w - ts)
            img = img[y:y + ts, x:x + ts]
            mask = mask[y:y + ts, x:x + ts]
            aug = self.geo(image=img, mask=mask)
            img, mask = aug["image"], aug["mask"]
        else:
            img = img[:ts, :ts]
            mask = mask[:ts, :ts]

        onehot = np.zeros((4, ts, ts), dtype=np.float32)
        for c in range(4):
            onehot[c] = (mask == c).astype(np.float32)
        img_t = torch.from_numpy(img.astype(np.float32).transpose(2, 0, 1)) / 127.5 - 1.0
        return torch.from_numpy(onehot), img_t
