#!/usr/bin/env python
"""tests/test_honest_metrics.py — tests CPU (sin pytest) del kit honesto.

Cubre: cobertura de ventanas, pesos Hann, Dice/IoU con convencion de clases
ausentes, HD/HD95 y el flujo completo de blending con un modelo dummy.

Uso:  ~/miniconda3/envs/mioaug/bin/python tests/test_honest_metrics.py
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

from eval_fullimage import (window_starts, hann2d, class_dice_iou, hd_pair,
                            infer_full_image, MEAN, STD)  # noqa: E402


def test_window_starts():
    starts = window_starts(4000, 512, 256)
    assert starts[0] == 0 and starts[-1] == 4000 - 512, starts
    cov = np.zeros(4000, dtype=int)
    for s in starts:
        cov[s:s + 512] += 1
    assert cov.min() >= 1, "hay pixeles sin cubrir"
    assert window_starts(512, 512, 256) == [0]
    assert window_starts(700, 512, 512) == [0, 188]
    assert window_starts(6000, 512, 384)[0] == 0
    assert window_starts(6000, 512, 384)[-1] == 6000 - 512
    print("OK window_starts")


def test_hann2d():
    w = hann2d(512)
    assert w.shape == (512, 512)
    assert w.min() > 0 and w.max() <= 1.0
    assert abs(w[256, 256] - 1.0) < 1e-3 or w[256, 256] > 0.99
    print("OK hann2d")


def test_class_dice_iou():
    a = np.zeros((10, 10), np.uint8)
    b = np.zeros((10, 10), np.uint8)
    a[2:8, 2:8] = 1
    assert class_dice_iou(a, a.copy(), 1) == (1.0, 1.0)
    assert class_dice_iou(b, b.copy(), 1) is None          # ausente en ambos
    assert class_dice_iou(a, b, 1) == (0.0, 0.0)           # pred sin GT (FP)
    assert class_dice_iou(b, a, 1) == (0.0, 0.0)           # GT sin pred
    p = np.zeros((10, 10), np.uint8)
    p[4:8, 4:8] = 1
    d, i = class_dice_iou(p, a, 1)
    assert abs(d - 2 * 16 / (36 + 16)) < 1e-12, d
    assert abs(i - 16 / (36 + 16 - 16)) < 1e-12, i
    print("OK class_dice_iou")


def test_hd_pair():
    g = np.zeros((64, 64), bool)
    g[16:48, 16:48] = True
    hd, hd95 = hd_pair(g.copy(), g.copy())
    assert hd == 0.0 and hd95 == 0.0
    p2 = np.zeros((64, 64), bool)
    p2[19:51, 16:48] = True
    hd, hd95 = hd_pair(p2, g)
    assert 0 < hd95 <= 3.5, hd95
    assert hd <= 3.5, hd
    assert hd_pair(np.zeros((64, 64), bool), g) is None    # pred vacia
    print("OK hd_pair")


class DummyModel(torch.nn.Module):
    """Pinta clase 1 donde el pixel original es rojo (R/255 > 0.5)."""

    def forward(self, x):
        r = x[:, 0:1] * float(STD[0]) + float(MEAN[0])
        red = (r > 0.5).float()
        z = torch.zeros_like(red)
        return torch.cat([1.0 - red, red, z, z], dim=1)


def test_blend_end_to_end():
    img = np.zeros((640, 640, 3), np.uint8)
    img[200:440, 200:440] = (255, 0, 0)
    model = DummyModel()
    pred = infer_full_image(model, None, 0.5, img, tile=256, stride=128,
                            batch=4, device="cpu", amp_dtype=None, tta="none",
                            verbose=False)
    exp = np.zeros((640, 640), np.uint8)
    exp[200:440, 200:440] = 1
    assert pred.shape == (640, 640)
    assert np.array_equal(pred, exp), "el blending no reproduce el recuadro"
    print("OK blend end-to-end")


def test_tta_shapes():
    img = np.zeros((320, 320, 3), np.uint8)
    img[100:220, 100:220] = (255, 0, 0)
    model = DummyModel()
    for tta in ("flips", "d4"):
        pred = infer_full_image(model, None, 0.5, img, tile=256, stride=128,
                                batch=2, device="cpu", amp_dtype=None, tta=tta,
                                verbose=False)
        exp = np.zeros((320, 320), np.uint8)
        exp[100:220, 100:220] = 1
        assert np.array_equal(pred, exp), f"TTA {tta} rompe la prediccion"
    print("OK tta flips/d4")


if __name__ == "__main__":
    test_window_starts()
    test_hann2d()
    test_class_dice_iou()
    test_hd_pair()
    test_blend_end_to_end()
    test_tta_shapes()
    print("TODOS LOS TESTS OK")
