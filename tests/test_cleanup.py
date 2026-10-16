#!/usr/bin/env python
"""Tests unitarios de src/cleanup.py (podado de islas por componentes)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from cleanup import cleanup_pred  # noqa: E402


def disk(shape, cy, cx, r, val):
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    m = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
    out = np.zeros(shape, np.uint8)
    out[m] = val
    return out


def main():
    H = W = 512
    checks = []

    # 1) identidad con parametros desactivados
    pred = np.zeros((64, 64), np.uint8)
    pred[10:20, 10:20] = 1
    pred[0:2, 62:64] = 3
    p2 = cleanup_pred(pred, close_r=0, min_cluster_px=0, max_dist=None)
    checks.append(("identidad desactivada", np.array_equal(pred, p2)))

    # 2) isla lejana eliminada; region grande intacta; clases preservadas
    big = disk((H, W), 256, 256, 60, 1)
    big[250:270, 240:280] = 2
    big[230:240, 300:312] = 3
    island = disk((H, W), 40, 40, 3, 3)  # ~29 px
    pred = np.maximum(big, island)
    p2 = cleanup_pred(pred, close_r=2, min_cluster_px=400)
    checks.append(("isla lejana eliminada", p2[40, 40] == 0))
    checks.append(("region grande intacta",
                   np.array_equal(p2[180:340, 170:350], pred[180:340, 170:350])))
    checks.append(("valores de clase preservados",
                   set(np.unique(p2).tolist()) == set(np.unique(pred).tolist())))

    # 3) fragmento cercano: sin max_dist se va, con max_dist vuelve
    frag = disk((H, W), 256, 340, 2, 1)  # borde del blob en x=316; gap ~21 px
    pred2 = np.maximum(big.copy(), frag)
    p_no = cleanup_pred(pred2, close_r=2, min_cluster_px=400, max_dist=None)
    p_dist = cleanup_pred(pred2, close_r=2, min_cluster_px=400, max_dist=64)
    checks.append(("fragmento chico se va sin max_dist", p_no[256, 340] == 0))
    checks.append(("fragmento cercano vuelve con max_dist", p_dist[256, 340] == 1))
    checks.append(("blob intacto con max_dist",
                   np.array_equal(p_dist[180:340, 170:350], pred2[180:340, 170:350])))

    # 4) nada suficientemente grande -> se conserva la mayor (una sola)
    pred3 = np.zeros((H, W), np.uint8)
    pred3[10:13, 10:13] = 1   # 9 px, label 1 (orden de escaneo)
    pred3[400:403, 400:403] = 2  # 9 px, label 2
    p3 = cleanup_pred(pred3, close_r=0, min_cluster_px=1000)
    checks.append(("no se borra todo", int(p3.sum()) > 0))
    checks.append(("conserva solo la mayor (1a en empate)",
                   p3[11, 11] == 1 and p3[401, 401] == 0))

    # 5) pred vacia -> vacia
    p5 = cleanup_pred(np.zeros((32, 32), np.uint8), close_r=2, min_cluster_px=100)
    checks.append(("pred vacia -> vacia", int(p5.sum()) == 0))

    # 6) dos clusters grandes: ambos se conservan
    pred6 = np.zeros((H, W), np.uint8)
    pred6 = np.maximum(pred6, disk((H, W), 100, 100, 40, 2))
    pred6 = np.maximum(pred6, disk((H, W), 380, 380, 40, 3))
    p6 = cleanup_pred(pred6, close_r=2, min_cluster_px=400)
    checks.append(("dos clusters grandes conservados",
                   p6[100, 100] == 2 and p6[380, 380] == 3))

    ok = all(v for _, v in checks)
    for name, v in checks:
        print(f"  [{'OK' if v else 'FALLO'}] {name}")
    print("TEST cleanup:", "TODO OK" if ok else "HAY FALLOS")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
