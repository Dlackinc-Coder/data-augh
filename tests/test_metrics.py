#!/usr/bin/env python
"""tests/test_metrics.py — tests de dice_multiclass/iou_multiclass (src/eval.py).

Verifica la correccion del arreglo del 9-oct-2026 (clases ausentes en GT y en
pred se OMITEN del promedio; antes contaban como Dice=1.0 y inflaban la media).

Uso:  ~/miniconda3/envs/mioaug/bin/python tests/test_metrics.py
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

from eval import dice_multiclass, iou_multiclass  # noqa: E402


def test_perfect():
    logits = torch.zeros(1, 4, 8, 8)
    mask = torch.zeros(1, 8, 8, dtype=torch.long)
    mask[:, :4] = 1
    logits[:, 1, :4] = 10.0
    logits[:, 0, 4:] = 10.0
    d = dice_multiclass(logits, mask)
    i = iou_multiclass(logits, mask)
    assert abs(d - 1.0) < 1e-6, d
    assert abs(i - 1.0) < 1e-6, i
    print("OK perfecto = 1.0 (clases 2,3 ausentes en ambos se omiten)")


def test_absent_both_not_promoted():
    """El caso que distinguia al bug: clases ausentes en ambos NO cuentan 1.0."""
    logits = torch.zeros(1, 4, 8, 8)
    mask = torch.zeros(1, 8, 8, dtype=torch.long)
    mask[:, 2:6, 2:6] = 3                    # solo clases 0 y 3
    logits[:, 3, 3:7, 2:6] = 10.0            # pred clase 3 desplazada 1 fila
    logits[:, 0, :3] = 5.0                   # resto: clase 0 (aproximada)
    logits[:, 0, 7:] = 5.0
    logits[:, 0, 3:7, :2] = 5.0
    logits[:, 0, 3:7, 6:] = 5.0
    logits[:, 0, 3:7, 2:6] = -5.0            # que la clase 3 gane su zona
    d = dice_multiclass(logits, mask)
    # nuevo: media(0.9167 [c0], 0.75 [c3]) = 0.8333
    # viejo (bug): media(0.9167, 1, 1, 0.75) = 0.9167
    assert abs(d - 0.8333) < 0.01, d
    assert d < 0.9, f"dice {d} parece inflado por clases ausentes"
    print(f"OK ausentes-no-infladas: dice = {d:.4f} (esperado 0.8333)")


def test_gt_absent_pred_present():
    """Clase predicha donde el GT no la tiene: cuenta 0 (no se omite)."""
    logits = torch.zeros(1, 4, 8, 8)
    mask = torch.zeros(1, 8, 8, dtype=torch.long)   # todo fondo
    logits[:, 0] = 5.0
    logits[:, 2, :2, :2] = 10.0                      # FP de clase 2
    d = dice_multiclass(logits, mask)
    # clase 0: dice ~0.968; clase 2: ~0 -> media ~0.484
    assert 0.40 < d < 0.55, d
    print(f"OK FP penalizado: dice = {d:.4f}")


def test_all_background_perfect():
    logits = torch.zeros(1, 4, 8, 8)
    logits[:, 0] = 10.0                              # pred: todo fondo
    mask = torch.zeros(1, 8, 8, dtype=torch.long)    # GT: todo fondo
    d = dice_multiclass(logits, mask)
    # tras argmax la clase 0 siempre esta presente; aqui pred=gt=fondo -> 1.0
    assert d == 1.0, f"fondo perfecto => 1.0, dio {d}"
    print("OK fondo perfecto = 1.0")


if __name__ == "__main__":
    test_perfect()
    test_absent_both_not_promoted()
    test_gt_absent_pred_present()
    test_all_background_perfect()
    print("TODOS LOS TESTS DE METRICA OK")
