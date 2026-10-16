#!/usr/bin/env python
"""cleanup.py — podado de islas por componentes para predicciones full-image.

Motivacion (16-oct-2026): las preds full-image de los modelos contienen
"islas" lejanas al tejido (falsos positivos) que disparan HD/HD95 (300-750 px)
y meten ruido en Dice. Regla: se cierra morfologicamente el primer plano
(para fusionar speckle fino), se etiquetan sus componentes y se descartan las
mas pequenas que min_cluster_px. Opcionalmente, tambien se descartan pixeles
muy lejanos (max_dist) a cualquier componente conservada.

Calibracion con GT (test, 37 imgs, 16-oct):
  - los cortes reales del corazon son clusters grandes (miles-px);
  - el GT tiene speckle de anotacion (~p50 24 px tras cierre r=2) cerca de los
    cortes; por eso la poda se hace sobre PREDICCIONES (suaves) y con
    min_cluster_px bastante por debajo del corte real mas pequeno.

La funcion nunca borra todo: si ningun cluster alcanza min_cluster_px se
conserva solo la componente mayor.
"""
import numpy as np
from scipy import ndimage as ndi


def cleanup_pred(pred, close_r=2, min_cluster_px=2000, max_dist=None):
    """Devuelve `pred` con las islas pequenas/lejanas puestas a 0 (bg).

    Args:
      pred: uint8 HxW con clases 0..3.
      close_r: radio del cierre morfologico previo a etiquetar (0 = sin cierre).
      min_cluster_px: tamano minimo (px) de un cluster para conservarlo.
      max_dist: si no es None, los pixeles de primer plano a mas de
        max_dist px de cualquier cluster conservado tambien se eliminan.

    No-op exacto si close_r <= 0 y min_cluster_px <= 0 y max_dist is None.
    """
    pred = np.asarray(pred, dtype=np.uint8)
    if close_r <= 0 and min_cluster_px <= 0 and max_dist is None:
        return pred.copy()
    U = pred > 0
    if not U.any():
        return pred.copy()

    Uc = ndi.binary_closing(U, iterations=int(close_r)) if close_r > 0 else U
    lab, n = ndi.label(Uc)
    if n == 0:
        return pred.copy()

    sizes = np.bincount(lab.ravel(), minlength=n + 1).astype(np.int64)
    sizes[0] = 0
    keep_ids = np.where(sizes >= max(int(min_cluster_px), 1))[0]
    keep_ids = keep_ids[keep_ids > 0]
    if keep_ids.size == 0:
        keep_ids = np.array([int(np.argmax(sizes))])
    keep = np.isin(lab, keep_ids)

    if max_dist is not None:
        dt = ndi.distance_transform_edt(~keep)
        keep = keep | (U & (dt <= float(max_dist)))

    out = np.zeros_like(pred)
    sel = U & keep
    out[sel] = pred[sel]
    return out
