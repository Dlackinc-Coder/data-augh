#!/usr/bin/env python
"""Tests unitarios de las perdidas de src/train_seg.py (Tversky, Fase 3)."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from train_seg import tversky_loss, tversky_ce_loss  # noqa: E402


def main():
    checks = []

    # 1) alpha=beta=0.5 == soft dice multiclase (calculo manual)
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 32, 32)
    targets = torch.randint(0, 4, (2, 32, 32))
    tv = float(tversky_loss(logits, targets, 0.5, 0.5))
    probs = torch.softmax(logits, 1)
    exp = 0.0
    for c in range(4):
        p = probs[:, c]
        g = (targets == c).float()
        tp = (p * g).sum()
        fp = (p * (1 - g)).sum()
        fn = ((1 - p) * g).sum()
        exp = exp + 1 - (tp + 1e-6) / (tp + 0.5 * fp + 0.5 * fn + 1e-6)
    exp = float(exp / 4)
    checks.append(("alpha=beta=0.5 == soft dice",
                   abs(tv - exp) < 1e-6))

    # 2) FN-heavy en clase 1: beta mayor -> mayor perdida (favorece recall)
    t2 = torch.zeros(1, 16, 16, dtype=torch.long)
    t2[0, 0, :10] = 1  # 10 px de clase 1 (fila 0)
    lg = torch.full((1, 4, 16, 16), -8.0)
    lg[0, 0] = 8.0  # bg casi seguro
    lg[0, 1, 0, :10] = 8.0 + float(torch.log(torch.tensor(0.4 / 0.6)))  # p1~0.4
    lo_beta = float(tversky_loss(lg, t2, alpha=[0.3] * 4, beta=[0.5, 0.2, 0.5, 0.5]))
    hi_beta = float(tversky_loss(lg, t2, alpha=[0.3] * 4, beta=[0.5, 0.8, 0.5, 0.5]))
    checks.append(("FN-heavy: beta 0.8 > beta 0.2 en perdida", hi_beta > lo_beta))

    # 3) FP-heavy en clase 1: alpha mayor -> mayor perdida
    t3 = torch.zeros(1, 16, 16, dtype=torch.long)
    t3[0, 0, :10] = 1
    lg3 = torch.full((1, 4, 16, 16), -8.0)
    lg3[0, 0] = 8.0
    lg3[0, 1, 0, :10] = 8.0                     # aciertos perfectos clase 1
    lg3[0, 1, 1, :20] = 8.0 + float(torch.log(torch.tensor(0.4 / 0.6)))  # ~0.4 en 20 px de bg
    lo_a = float(tversky_loss(lg3, t3, alpha=[0.3, 0.2, 0.3, 0.3], beta=[0.5] * 4))
    hi_a = float(tversky_loss(lg3, t3, alpha=[0.3, 0.8, 0.3, 0.3], beta=[0.5] * 4))
    checks.append(("FP-heavy: alpha 0.8 > alpha 0.2 en perdida", hi_a > lo_a))

    # 4) tversky_ce == tv_weight*tv + ce_weight*ce (exacto)
    torch.manual_seed(1)
    lg4 = torch.randn(2, 4, 24, 24)
    t4 = torch.randint(0, 4, (2, 24, 24))
    tv4 = float(tversky_loss(lg4, t4, 0.3, 0.7))
    ce4 = float(torch.nn.functional.cross_entropy(lg4, t4))
    comb = float(tversky_ce_loss(lg4, t4, 0.3, 0.7, tv_weight=1.0, ce_weight=0.5))
    checks.append(("tversky_ce = tv + 0.5*ce", abs(comb - (tv4 + 0.5 * ce4)) < 1e-6))

    # 5) gradientes finitos y no nulos
    lg5 = torch.randn(1, 4, 16, 16, requires_grad=True)
    t5 = torch.randint(0, 4, (1, 16, 16))
    loss = tversky_loss(lg5, t5, 0.3, 0.7)
    loss.backward()
    g = lg5.grad
    checks.append(("grad finito y no nulo",
                   g is not None and torch.isfinite(g).all().item()
                   and float(g.abs().sum()) > 0))

    # 6) prediccion perfecta (logits muy picos) -> perdida ~0
    t6 = torch.zeros(1, 8, 8, dtype=torch.long)
    t6[0, :4] = 1
    t6[0, 4:] = 2
    lg6 = torch.full((1, 4, 8, 8), -12.0)
    lg6[0, 1, :4] = 12.0
    lg6[0, 2, 4:] = 12.0
    p6 = float(tversky_loss(lg6, t6, 0.3, 0.7))
    checks.append(("prediccion casi perfecta -> perdida < 1e-3", p6 < 1e-3))

    ok = all(v for _, v in checks)
    for name, v in checks:
        print(f"  [{'OK' if v else 'FALLO'}] {name}")
    print("TEST losses:", "TODO OK" if ok else "HAY FALLOS")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
