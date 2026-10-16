#!/usr/bin/env python
"""gt_floor.py — ancla de HD por speckle de anotacion en el GT (CPU, solo lectura).

Para cada imagen y clase: GT completa vs GT "limpia" (componentes < 800 px del
primer plano cerrado r=2 eliminadas). hd_pair entre ambas da una cota inferior
de HD/HD95 para cualquier prediccion suave que no reproduzca los speckles de
anotacion (explica el piso de HD ~200 px que queda tras la limpieza de islas).
"""
import json
import os
import sys

import cv2
import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, os.path.expanduser("~/projects/data-augh/src"))
from eval_fullimage import hd_pair, FG_CLASSES  # noqa: E402

root = os.path.expanduser("~/projects/data-augh/data/mouse_cardiac")
out = {}
for split in ("test", "val"):
    md = os.path.join(root, split, "mask")
    files = sorted(os.listdir(md))
    acc = {c: [] for c in FG_CLASSES}
    for f in files:
        m = cv2.imread(os.path.join(md, f), cv2.IMREAD_GRAYSCALE)
        m = np.clip(np.rint(m.astype(np.float32)), 0, 3).astype(np.uint8)
        U = m > 0
        Uc = ndi.binary_closing(U, iterations=2)
        lab, n = ndi.label(Uc)
        sizes = np.bincount(lab.ravel(), minlength=n + 1)
        sizes[0] = 0
        keep_ids = np.where(sizes >= 800)[0]
        keep_ids = keep_ids[keep_ids > 0]
        keep = np.isin(lab, keep_ids) if keep_ids.size else (Uc > 0)
        for c in FG_CLASSES:
            g_full = m == c
            if not g_full.any():
                continue
            g_clean = g_full & keep
            hd = hd_pair(g_full, g_clean)
            if hd is not None:
                acc[c].append(hd)
    out[split] = {c: {
        "n": len(acc[c]),
        "hd_mean": (float(np.mean([x[0] for x in acc[c]])) if acc[c] else None),
        "hd95_mean": (float(np.mean([x[1] for x in acc[c]])) if acc[c] else None),
    } for c in FG_CLASSES}
print(json.dumps(out, indent=1))
