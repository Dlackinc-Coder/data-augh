import os
import random
import numpy as np
import torch
import torch.utils.data as data
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


class TileDataset(data.Dataset):
    def __init__(self, root, split, tile_size, stride, tiles_per_image=8, augment=False, seed=42,
                 sampling_strategy="random", min_tissue_ratio=0.05, exploration_rate=0.1,
                 max_attempts=20, jitter_px=32):
        self.root = os.path.join(root, split)
        self.tile_size = tile_size
        self.stride = stride
        self.tiles_per_image = tiles_per_image
        self.augment = augment
        self.seed = seed
        self.sampling_strategy = sampling_strategy
        self.min_tissue_ratio = min_tissue_ratio
        self.exploration_rate = exploration_rate
        self.max_attempts = max_attempts
        self.jitter_px = jitter_px
        self.prob_maps = None
        
        self.images_dir = os.path.join(self.root, "img")
        self.masks_dir = os.path.join(self.root, "mask")
        
        self.image_files = sorted([f for f in os.listdir(self.images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        self.mask_files = sorted([f for f in os.listdir(self.masks_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        
        assert len(self.image_files) == len(self.mask_files), "Mismatch images/masks"
        
        if sampling_strategy == "guided":
            self.prob_maps = self._build_prob_maps()
        
        if augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Affine(translate_percent=0.1, scale=(0.8, 1.2), rotate=(-15, 15), p=0.7),
                A.GaussNoise(std_range=(0.01, 0.03), p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        
        random.seed(seed)
        np.random.seed(seed)
    
    def _build_prob_maps(self):
        prob_maps = []
        cell = max(32, self.tile_size // 4)
        for mf in self.mask_files:
            m = cv2.imread(os.path.join(self.masks_dir, mf), cv2.IMREAD_GRAYSCALE)
            if m is None:
                prob_maps.append(None)
                continue
            tissue = (m > 0).astype(np.float32)
            h, w = tissue.shape
            hc, wc = max(1, h // cell), max(1, w // cell)
            grid = tissue[:hc * cell, :wc * cell].reshape(hc, cell, wc, cell).mean(axis=(1, 3))
            grid = cv2.GaussianBlur(grid, (5, 5), 2).astype(np.float64)
            s = grid.sum()
            grid = grid / s if s > 1e-8 else None
            prob_maps.append(grid)
        return prob_maps
    
    def _guided_sample(self, img_idx, mask, max_y, max_x):
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
                p = p / p.sum()
                idx = np.random.choice(p.size, p=p)
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
            ys, xs = np.nonzero(tile > 0)
            if len(ys) == 0:
                return y, x
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
        tile_idx = idx % self.tiles_per_image
        
        img_path = os.path.join(self.images_dir, self.image_files[img_idx])
        mask_path = os.path.join(self.masks_dir, self.mask_files[img_idx])
        
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        h, w = img.shape[:2]
        max_y = h - self.tile_size
        max_x = w - self.tile_size
        
        if max_y <= 0 or max_x <= 0:
            raise ValueError(f"Image {img_path} smaller than tile_size")
        
        random.seed(self.seed + idx)
        np.random.seed(self.seed + idx)
        
        if self.sampling_strategy == "guided":
            y, x = self._guided_sample(img_idx, mask, max_y, max_x)
        else:
            y = random.randint(0, max_y)
            x = random.randint(0, max_x)
        
        img_tile = img[y:y+self.tile_size, x:x+self.tile_size]
        mask_tile = mask[y:y+self.tile_size, x:x+self.tile_size]
        
        mask_tile = np.clip(np.rint(mask_tile.astype(np.float32)), 0, 3).astype(np.uint8)
        
        if self.transform:
            out = self.transform(image=img_tile, mask=mask_tile)
            img_t = out["image"]
            mask_t = out["mask"].long()
        else:
            img_t = torch.from_numpy(img_tile.transpose(2, 0, 1)).float() / 255.0
            mask_t = torch.from_numpy(mask_tile).long()
        
        return img_t, mask_t


def make_loaders(cfg):
    ds_cfg = cfg["dataset"]
    train_cfg = cfg["train"]
    common = dict(root=ds_cfg["root"], tile_size=ds_cfg["tile_size"],
                  stride=ds_cfg.get("stride", ds_cfg["tile_size"]))
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
    train = TileDataset(**common, split="train", augment=True,
                        tiles_per_image=tiles, seed=seed, **sampling_kwargs)
    val = TileDataset(**common, split="val", augment=False,
                      tiles_per_image=tiles, seed=seed, **sampling_kwargs)
    test = TileDataset(**common, split="test", augment=False,
                       tiles_per_image=tiles, seed=seed, **sampling_kwargs)
    n_workers = min(4, os.cpu_count() or 1)
    train_dl = torch.utils.data.DataLoader(
        train, batch_size=train_cfg["batch_size"], shuffle=True,
        num_workers=train_cfg.get("num_workers", 4), pin_memory=True, drop_last=True)
    val_dl = torch.utils.data.DataLoader(
        val, batch_size=train_cfg["batch_size"], shuffle=False,
        num_workers=train_cfg.get("num_workers", 4), pin_memory=True)
    test_dl = torch.utils.data.DataLoader(
        test, batch_size=train_cfg["batch_size"], shuffle=False,
        num_workers=train_cfg.get("num_workers", 4), pin_memory=True)
    return train_dl, val_dl, test_dl
