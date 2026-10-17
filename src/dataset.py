"""Tile dataset for mouse cardiac slice segmentation.

Crops (tile_size x tile_size) tiles from 6000x4000 images/masks. Supports two
sampling strategies:

- ``random``: uniform random tile position.
- ``guided``: tile positions sampled from a tissue-density probability map
  (with exploration/jitter), so training sees more foreground tissue.
"""

import math
import os
import random

import cv2
import numpy as np
import torch
import torch.utils.data as data
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)
IMG_EXTS = (".jpg", ".jpeg", ".png")


def _list_index(dir_path):
    """Map stem -> filename for all image files in dir_path."""
    return {
        os.path.splitext(f)[0]: f
        for f in sorted(os.listdir(dir_path))
        if f.lower().endswith(IMG_EXTS)
    }


class TileDataset(data.Dataset):
    """Dataset of tiles sampled from full-size cardiac images.

    Pairs images/masks by filename stem (not by list position), decodes only
    the tiles it needs, and applies conservative geometric/photometric
    augmentation when ``augment=True``.
    """

    def __init__(self, root, split, tile_size, stride=None, tiles_per_image=8,
                 augment=False, seed=42, sampling_strategy="random",
                 min_tissue_ratio=0.05, exploration_rate=0.1, max_attempts=20,
                 jitter_px=32, num_classes=4,
                 mean=DEFAULT_MEAN, std=DEFAULT_STD):
        self.root = os.path.join(root, split)
        self.tile_size = tile_size
        self.tiles_per_image = tiles_per_image
        self.augment = augment
        self.seed = seed
        self.sampling_strategy = sampling_strategy
        self.min_tissue_ratio = min_tissue_ratio
        self.exploration_rate = exploration_rate
        self.max_attempts = max_attempts
        self.jitter_px = jitter_px
        self.num_classes = num_classes
        self.prob_maps = None

        self.images_dir = os.path.join(self.root, "img")
        self.masks_dir = os.path.join(self.root, "mask")

        imgs = _list_index(self.images_dir)
        msks = _list_index(self.masks_dir)
        common = sorted(set(imgs) & set(msks))
        only_img = sorted(set(imgs) - set(msks))
        only_mask = sorted(set(msks) - set(imgs))
        if only_img:
            print(f"[TileDataset] WARNING: {len(only_img)} images without mask in "
                  f"{self.root}: {only_img[:5]}{'...' if len(only_img) > 5 else ''}")
        if only_mask:
            print(f"[TileDataset] WARNING: {len(only_mask)} masks without image in "
                  f"{self.root}: {only_mask[:5]}{'...' if len(only_mask) > 5 else ''}")
        if not common:
            raise FileNotFoundError(f"No image/mask pairs found in {self.root}")
        self.image_files = [imgs[s] for s in common]
        self.mask_files = [msks[s] for s in common]

        if sampling_strategy == "guided":
            self.prob_maps = self._build_prob_maps()

        transforms = []
        if augment:
            transforms += [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Affine(translate_percent=0.1, scale=(0.8, 1.2), rotate=(-15, 15), p=0.7),
                A.GaussNoise(std_range=(0.01, 0.03), p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            ]
        transforms += [
            A.Normalize(mean=tuple(mean), std=tuple(std)),
            ToTensorV2(),
        ]
        self.transform = A.Compose(transforms)

    def _build_prob_maps(self):
        """Grid probability map of tissue density per mask (padded to full size)."""
        prob_maps = []
        cell = max(32, self.tile_size // 4)
        for mf in self.mask_files:
            m = cv2.imread(os.path.join(self.masks_dir, mf), cv2.IMREAD_GRAYSCALE)
            if m is None:
                prob_maps.append(None)
                continue
            tissue = (m > 0).astype(np.float32)
            h, w = tissue.shape
            hc, wc = max(1, math.ceil(h / cell)), max(1, math.ceil(w / cell))
            padded = np.zeros((hc * cell, wc * cell), dtype=np.float32)
            padded[:h, :w] = tissue
            grid = padded.reshape(hc, cell, wc, cell).mean(axis=(1, 3))
            grid = cv2.GaussianBlur(grid, (5, 5), 2).astype(np.float32)
            s = grid.sum()
            prob_maps.append(grid / s if s > 1e-8 else None)
        return prob_maps

    def _guided_sample(self, img_idx, mask, max_y, max_x):
        """Sample a tile position biased towards tissue, with jitter/exploration."""
        ts = self.tile_size
        pmap = None
        if self.prob_maps and img_idx < len(self.prob_maps):
            pmap = self.prob_maps[img_idx]
        for _ in range(self.max_attempts):
            if pmap is None or random.random() < self.exploration_rate:
                y = random.randint(0, max_y)
                x = random.randint(0, max_x)
            else:
                p = pmap.ravel()
                s = p.sum()
                if not np.isfinite(s) or s <= 0:
                    y = random.randint(0, max_y)
                    x = random.randint(0, max_x)
                else:
                    idx = np.random.choice(p.size, p=p / s)
                    cy, cx = np.unravel_index(idx, pmap.shape)
                    h, w = mask.shape
                    py = (cy + random.random()) * h / pmap.shape[0]
                    px = (cx + random.random()) * w / pmap.shape[1]
                    y = int(py) - ts // 2 + random.randint(-self.jitter_px, self.jitter_px)
                    x = int(px) - ts // 2 + random.randint(-self.jitter_px, self.jitter_px)
                    y = int(np.clip(y, 0, max_y))
                    x = int(np.clip(x, 0, max_x))
            tile = mask[y:y + ts, x:x + ts]
            if (tile > 0).mean() < self.min_tissue_ratio:
                continue
            # Refine towards the tissue centroid.
            ys, xs = np.nonzero(tile > 0)
            sy = int(ys.mean()) - ts // 2 + random.randint(-self.jitter_px, self.jitter_px)
            sx = int(xs.mean()) - ts // 2 + random.randint(-self.jitter_px, self.jitter_px)
            ny = int(np.clip(y + sy, 0, max_y))
            nx = int(np.clip(x + sx, 0, max_x))
            nt = mask[ny:ny + ts, nx:nx + ts]
            if (nt > 0).mean() >= self.min_tissue_ratio * 0.5:
                return ny, nx
            return y, x
        return random.randint(0, max_y), random.randint(0, max_x)

    def __len__(self):
        return len(self.image_files) * self.tiles_per_image

    def __getitem__(self, idx):
        img_idx = idx // self.tiles_per_image

        img_path = os.path.join(self.images_dir, self.image_files[img_idx])
        mask_path = os.path.join(self.masks_dir, self.mask_files[img_idx])

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not read mask: {mask_path}")
        if mask.shape[:2] != img.shape[:2]:
            raise ValueError(
                f"Image/mask size mismatch for {img_path}: "
                f"{img.shape[:2]} vs {mask.shape[:2]}")

        h, w = img.shape[:2]
        max_y = h - self.tile_size
        max_x = w - self.tile_size
        if max_y < 0 or max_x < 0:
            raise ValueError(
                f"Image {img_path} ({h}x{w}) smaller than tile_size {self.tile_size}")

        if self.sampling_strategy == "guided":
            y, x = self._guided_sample(img_idx, mask, max_y, max_x)
        else:
            y = random.randint(0, max_y)
            x = random.randint(0, max_x)

        img_tile = img[y:y + self.tile_size, x:x + self.tile_size]
        mask_tile = mask[y:y + self.tile_size, x:x + self.tile_size]

        mask_tile = np.clip(np.rint(mask_tile.astype(np.float32)),
                            0, self.num_classes - 1).astype(np.uint8)

        out = self.transform(image=img_tile, mask=mask_tile)
        return out["image"], out["mask"].long()


def make_loaders(cfg, need_test=True):
    """Build (train, val[, test]) DataLoaders from a config dict.

    Workers are seeded by PyTorch per epoch (no manual global reseeding), so
    augmentation varies across epochs. When ``need_test=False`` the test
    split is not constructed (avoids wasted mask parsing during training).
    """
    ds_cfg = cfg["dataset"]
    train_cfg = cfg["train"]
    num_classes = cfg.get("model", {}).get("classes", 4)
    common = dict(root=ds_cfg["root"], tile_size=ds_cfg["tile_size"],
                  num_classes=num_classes,
                  mean=ds_cfg.get("mean", DEFAULT_MEAN),
                  std=ds_cfg.get("std", DEFAULT_STD))
    sampling = ds_cfg.get("sampling", {})
    sampling_kwargs = dict(
        sampling_strategy=sampling.get("strategy", "random"),
        min_tissue_ratio=sampling.get("min_tissue_ratio", 0.05),
        exploration_rate=sampling.get("exploration_rate", 0.1),
        max_attempts=sampling.get("max_attempts", 20),
        jitter_px=sampling.get("jitter_px", 32),
    )
    tiles = ds_cfg.get("tiles_per_image", 8)
    seed = cfg.get("seed", 42)
    num_workers = train_cfg.get("num_workers", 4)

    train = TileDataset(**common, split="train", augment=True,
                        tiles_per_image=tiles, seed=seed, **sampling_kwargs)
    val = TileDataset(**common, split="val", augment=False,
                      tiles_per_image=tiles, seed=seed, **sampling_kwargs)

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_dl = torch.utils.data.DataLoader(
        train, batch_size=train_cfg["batch_size"], shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        generator=generator)
    val_dl = torch.utils.data.DataLoader(
        val, batch_size=train_cfg["batch_size"], shuffle=False,
        num_workers=num_workers, pin_memory=True)

    if not need_test:
        return train_dl, val_dl, None

    test = TileDataset(**common, split="test", augment=False,
                       tiles_per_image=tiles, seed=seed, **sampling_kwargs)
    test_dl = torch.utils.data.DataLoader(
        test, batch_size=train_cfg["batch_size"], shuffle=False,
        num_workers=num_workers, pin_memory=True)
    return train_dl, val_dl, test_dl
