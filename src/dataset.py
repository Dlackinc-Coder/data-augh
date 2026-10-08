import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


class TileDataset(Dataset):
    def __init__(self, root, split="train", tile_size=512, stride=512,
                 augment=True, tiles_per_image=8, seed=42):
        self.root = os.path.join(root, split)
        self.img_dir = os.path.join(self.root, "img")
        self.mask_dir = os.path.join(self.root, "mask")
        self.tile_size = tile_size
        self.stride = stride
        self.tiles_per_image = tiles_per_image
        self.augment = augment
        self.seed = seed
        self.pairs = self._find_pairs()
        self.grid = None
        if not augment:
            self.grid = self._build_grid()

        if augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Affine(translate_percent=0.1, scale=(0.8, 1.2),
                         rotate=15, cval=0, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2,
                                           contrast_limit=0.2, p=0.5),
                A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30,
                                     val_shift_limit=20, p=0.3),
                A.GaussNoise(std_range=(0.02, 0.12), p=0.2),
            ])
        self.to_tensor = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    def _find_pairs(self):
        stems = sorted(
            f[:-len(os.path.splitext(f)[1])]
            for f in os.listdir(self.img_dir) if f.lower().endswith(IMG_EXTS)
        )
        pairs = []
        for s in stems:
            for ext in IMG_EXTS:
                if os.path.exists(os.path.join(self.mask_dir, s + ext)):
                    pairs.append((os.path.join(self.img_dir, s + ext),
                                  os.path.join(self.mask_dir, s + ext)))
                    break
        if not pairs:
            raise FileNotFoundError(f"no image/mask pairs in {self.root}")
        return pairs

    def _build_grid(self):
        grid = []
        img_pil = Image.open(self.pairs[0][0])
        h, w = img_pil.height, img_pil.width
        ts, st = self.tile_size, self.stride
        for y in range(0, max(h - ts, 0) + 1, st):
            for x in range(0, max(w - ts, 0) + 1, st):
                if y + ts > h:
                    y = h - ts
                if x + ts > w:
                    x = w - ts
                grid.append((y, x))
        return sorted(set(grid))

    @staticmethod
    def load_mask(path):
        m = np.array(Image.open(path))
        if m.ndim == 3:
            m = m[..., 0]
        m = np.clip(np.rint(m.astype(np.float32)), 0, 3).astype(np.uint8)
        return m

    def __len__(self):
        if self.augment:
            return len(self.pairs) * self.tiles_per_image
        return len(self.pairs) * len(self.grid)

    def __getitem__(self, idx):
        if self.augment:
            pair = self.pairs[idx % len(self.pairs)]
            img = np.array(Image.open(pair[0]).convert("RGB"))
            mask = self.load_mask(pair[1])
            h, w = mask.shape
            ts = self.tile_size
            y = random.randint(0, h - ts)
            x = random.randint(0, w - ts)
            img = img[y:y + ts, x:x + ts]
            mask = mask[y:y + ts, x:x + ts]
            if self.transform:
                aug = self.transform(image=img, mask=mask)
                img, mask = aug["image"], aug["mask"]
        else:
            pair = self.pairs[idx // len(self.grid)]
            y, x = self.grid[idx % len(self.grid)]
            ts = self.tile_size
            img = np.array(Image.open(pair[0]).convert("RGB"))[y:y + ts, x:x + ts]
            mask = self.load_mask(pair[1])[y:y + ts, x:x + ts]

        out = self.to_tensor(image=img)
        img_t = out["image"]
        mask_t = torch.from_numpy(mask.astype(np.int64))
        return img_t, mask_t


def make_loaders(cfg):
    ds_cfg = cfg["dataset"]
    common = dict(root=ds_cfg["root"], tile_size=ds_cfg["tile_size"],
                  stride=ds_cfg.get("stride", ds_cfg["tile_size"]))
    train = TileDataset(**common, split="train", augment=True,
                        tiles_per_image=ds_cfg.get("tiles_per_image", 8),
                        seed=cfg.get("seed", 42))
    val = TileDataset(**common, split="val", augment=False)
    test = TileDataset(**common, split="test", augment=False)
    n_workers = min(4, os.cpu_count() or 1)
    train_dl = torch.utils.data.DataLoader(
        train, batch_size=cfg["train"]["batch_size"], shuffle=True,
        num_workers=n_workers, pin_memory=True, drop_last=True)
    val_dl = torch.utils.data.DataLoader(
        val, batch_size=cfg["train"]["batch_size"], shuffle=False,
        num_workers=n_workers, pin_memory=True)
    test_dl = torch.utils.data.DataLoader(
        test, batch_size=cfg["train"]["batch_size"], shuffle=False,
        num_workers=n_workers, pin_memory=True)
    return train_dl, val_dl, test_dl
