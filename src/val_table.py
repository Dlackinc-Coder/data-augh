#!/usr/bin/env python
"""val_table.py — tabla de decision de protocolo en val (raw + limpieza offline).

Lee runs/eval_honest_val/*/metrics.json y runs/eval_cleanup_val/*.json y lista
las variantes ordenadas por AVG3 (desc). Uso:
  python src/val_table.py
"""
import glob
import json
import os

PROJECT = os.path.expanduser("~/projects/data-augh")

rows = []
for p in sorted(glob.glob(os.path.join(PROJECT, "runs/eval_honest_val/*/metrics.json"))):
    tag = os.path.basename(os.path.dirname(p))
    a = json.load(open(p))["aggregate"]
    rows.append((tag, "raw", a["avg3_mean"], a["avg_hd95"], a["avg_hd"]))

for p in sorted(glob.glob(os.path.join(PROJECT, "runs/eval_cleanup_val/*.json"))):
    d = json.load(open(p))
    tag = os.path.basename(p).replace(".json", "")
    if tag.endswith("_wider"):
        tag = tag[: -len("_wider")]
    for r in d["results"]:
        v = r["variant"]["label"]
        if v == "cl0_min0_dnone":
            continue
        a = r["aggregate"]
        rows.append((tag, v, a["avg3_mean"], a["avg_hd95"], a["avg_hd"]))

rows.sort(key=lambda r: (-(r[2] or 0)))
print(f"{'run':<22} {'variante':<18} {'avg3%':>7} {'HD95':>7} {'HD':>7}")
for tag, v, a3, h95, hd in rows:
    print(f"{tag:<22} {v:<18} {a3 * 100:>7.2f} {h95:>7.1f} {hd:>7.1f}")
