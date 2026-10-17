#!/usr/bin/env python
"""make_charts.py — gráficos PNG (Pillow) para la comparativa visual 16-oct-2026.

Uso (desde la raíz del proyecto):  python scripts/make_charts.py
Salida: docs/img/final_vs_ducnet.png, docs/img/camino.png, docs/img/panorama.png
"""
import os

from PIL import Image, ImageDraw, ImageFont

OUT = "docs/img"
os.makedirs(OUT, exist_ok=True)

BLUE = (46, 111, 219)
LIGHT_BLUE = (127, 168, 224)
GRAY = (154, 160, 166)
GREEN = (46, 158, 91)
ORANGE = (242, 140, 40)
LIGHT_ORANGE = (242, 179, 94)
RED = (219, 68, 55)
DARK = (32, 33, 36)
GRID = (222, 226, 230)

_TTF = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
_TTF_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def F(size, bold=False):
    for p in ([_TTF_B, _TTF] if bold else [_TTF]):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()


def tcenter(d, cx, y, s, font, fill=DARK):
    w = d.textlength(s, font=font)
    d.text((cx - w / 2, y), s, font=font, fill=fill)


def tright(d, x, y, s, font, fill=DARK):
    w = d.textlength(s, font=font)
    d.text((x - w, y), s, font=font, fill=fill)


def chart_final_vs_ducnet():
    W, H = 1100, 640
    L, R, T, B = 90, 40, 110, 80
    y0, y1 = 70.0, 89.0
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    ft, fl, fv = F(30, True), F(20), F(18, True)
    tcenter(d, W / 2, 30, "Resultado final en TEST — Dice (%)", ft)

    def ypx(v):
        return T + (y1 - v) / (y1 - y0) * (H - T - B)

    for v in [70, 75, 80, 85]:
        y = ypx(v)
        d.line([(L, y), (W - R, y)], fill=GRID, width=1)
        tright(d, L - 10, y - 12, str(v), fl, GRAY)
    d.line([(L, T - 20), (L, H - B)], fill=GRAY, width=2)
    d.line([(L, H - B), (W - R, H - B)], fill=GRAY, width=2)

    cats = ["LV", "AAR", "IS", "AVG3"]
    mio = [84.83, 75.70, 83.11, 81.21]
    duc = [84.22, 77.69, 80.13, 80.68]
    gw = (W - L - R) / len(cats)
    bw, gap = 80, 14
    for i, c in enumerate(cats):
        cx = L + gw * i + gw / 2
        xa = cx - bw - gap / 2
        xb = cx + gap / 2
        ya, yb = ypx(mio[i]), ypx(duc[i])
        d.rectangle([xa, ya, xa + bw, H - B], fill=BLUE)
        d.rectangle([xb, yb, xb + bw, H - B], fill=GRAY)
        tcenter(d, xa + bw / 2, ya - 30, f"{mio[i]:.2f}", fv, BLUE)
        tcenter(d, xb + bw / 2, yb - 30, f"{duc[i]:.2f}", fv, GRAY)
        tcenter(d, cx, H - B + 14, c, fl)

    d.rectangle([W - 360, 96, W - 30, 172], fill="white")
    d.rectangle([W - 350, 108, W - 330, 128], fill=BLUE)
    d.text((W - 322, 110), "MioClinic (final)", font=fl, fill=DARK)
    d.rectangle([W - 350, 140, W - 330, 160], fill=GRAY)
    d.text((W - 322, 142), "DUCNet 2026", font=fl, fill=DARK)
    img.save(f"{OUT}/final_vs_ducnet.png")
    print("ok final_vs_ducnet.png")


def chart_camino():
    W, H = 1000, 620
    L, R, T, B = 90, 40, 110, 90
    y0, y1 = 79.0, 82.0
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    ft, fl, fv = F(28, True), F(19), F(20, True)
    tcenter(d, W / 2, 30, "Camino al cierre — AVG3 % en test", ft)

    def ypx(v):
        return T + (y1 - v) / (y1 - y0) * (H - T - B)

    for v in [79, 80, 81, 82]:
        y = ypx(v)
        d.line([(L, y), (W - R, y)], fill=GRID, width=1)
        tright(d, L - 10, y - 12, str(v), fl, GRAY)
    d.line([(L, T - 20), (L, H - B)], fill=GRAY, width=2)
    d.line([(L, H - B), (W - R, H - B)], fill=GRAY, width=2)

    yd = ypx(80.68)
    for x in range(L, W - R, 18):
        d.line([(x, yd), (x + 9, yd)], fill=RED, width=3)
    d.text((L + 6, yd - 34), "DUCNet 80.68", font=F(18, True), fill=RED)

    items = [
        ("Mejor single", 80.23, LIGHT_BLUE),
        ("Single +TTA", 80.47, LIGHT_ORANGE),
        ("Ensemble", 81.15, BLUE),
        ("FINAL", 81.21, GREEN),
    ]
    gw = (W - L - R) / len(items)
    bw = 110
    for i, (lab, v, col) in enumerate(items):
        cx = L + gw * i + gw / 2
        xa = cx - bw / 2
        ya = ypx(v)
        d.rectangle([xa, ya, xa + bw, H - B], fill=col)
        tcenter(d, cx, ya - 34, f"{v:.2f}", fv, col)
        tcenter(d, cx, H - B + 16, lab, fl)
    img.save(f"{OUT}/camino.png")
    print("ok camino.png")


def chart_panorama():
    rows = []
    for lab, v in [("J (ViT-B)", 69.12), ("A-v3", 72.70), ("I (ConvNeXt)", 73.07),
                   ("A-v4", 76.22), ("H (PVTv2)", 77.55), ("A-v2", 77.59),
                   ("G-v1", 77.68), ("G-v2", 80.23), ("G-v3", 80.23)]:
        rows.append((lab, v, LIGHT_BLUE))
    for lab, v in [("A-v3@tta", 73.39), ("A-v4@tta", 76.61),
                   ("G-v3@tta", 80.30), ("G-v2@tta", 80.47)]:
        rows.append((lab, v, LIGHT_ORANGE))
    for lab, v in [("A-v3+G-v2", 80.89), ("A-v4+G-v2", 81.02),
                   ("A-v3+G-v3", 81.02), ("A-v4+G-v3", 81.15)]:
        rows.append((lab, v, BLUE))
    rows.append(("FINAL (A-v4@tta+G-v3@tta+limp)", 81.21, GREEN))

    W = 1280
    L, R, T = 300, 130, 130
    rh = 30
    H = T + len(rows) * rh + 50
    x0, x1 = L, W - R
    vmin, vmax = 68.0, 83.0

    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    ft, fl, fv = F(26, True), F(17), F(15, True)
    tcenter(d, W / 2, 24, "Panorama completo en test (protocolo honesto) — AVG3 %", ft)

    def xpx(v):
        return x0 + (v - vmin) / (vmax - vmin) * (x1 - x0)

    for v in range(68, 84, 2):
        x = xpx(v)
        d.line([(x, T - 14), (x, T + len(rows) * rh)], fill=GRID, width=1)
        tcenter(d, x, T + len(rows) * rh + 6, str(v), fl, GRAY)

    xd = xpx(80.68)
    y = T - 12
    while y < T + len(rows) * rh:
        d.line([(xd, y), (xd, min(y + 10, T + len(rows) * rh))], fill=RED, width=3)
        y += 18
    d.text((xd + 6, T - 40), "DUCNet 80.68", font=F(17, True), fill=RED)

    for i, (lab, v, col) in enumerate(rows):
        y = T + i * rh
        xb = xpx(v)
        d.rectangle([x0, y + 5, xb, y + rh - 6], fill=col)
        tright(d, x0 - 10, y + 5, lab, fl)
        d.text((xb + 8, y + 5), f"{v:.2f}", font=fv, fill=DARK)

    ly = 66
    lx = 300
    for lab, col in [("single", LIGHT_BLUE), ("+TTA", LIGHT_ORANGE),
                     ("ensemble", BLUE), ("FINAL", GREEN)]:
        d.rectangle([lx, ly, lx + 16, ly + 16], fill=col)
        d.text((lx + 22, ly - 2), lab, font=fl, fill=DARK)
        lx += 22 + int(d.textlength(lab, font=fl)) + 28
    img.save(f"{OUT}/panorama.png")
    print("ok panorama.png")


if __name__ == "__main__":
    chart_final_vs_ducnet()
    chart_camino()
    chart_panorama()
    print("charts listos en", OUT)
