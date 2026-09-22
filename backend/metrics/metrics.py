from __future__ import annotations
from statistics import mean, pstdev

def summarise(rows):
    if not rows:
        return {}
    keys = ["avg_wait","service_rate","cancelled","empty_driving_ratio","utilisation","earnings"]
    out = {k: mean(float(r[k]) for r in rows) for k in keys if k in rows[0]}
    if "avg_wait" in rows[0]: out["avg_wait_std"] = pstdev(float(r["avg_wait"]) for r in rows)
    return out
