#!/usr/bin/env python3
"""Make small SIMULATED files in the NASA C-MAPSS format so you can test SkyFix
without downloading anything.  NOT real NASA data.

    python make_sample_data.py                 # writes sample_data/train_FD001.txt etc.
    python make_sample_data.py --engines 60
"""
import argparse, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="sample_data")
ap.add_argument("--subset", default="FD001")
ap.add_argument("--engines", type=int, default=40)
ap.add_argument("--seed", type=int, default=7)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)
os.makedirs(a.out, exist_ok=True)

N_SENS = 21
ACTIVE = [1, 3, 4, 6, 10, 11, 14]            # sensors that drift as the engine wears (0-based into s1..s21)
BASE = rng.uniform(100, 600, N_SENS)
SCALE = rng.uniform(0.5, 5, N_SENS)
DIR = rng.choice([-1, 1], N_SENS)

MULTI = a.subset in ("FD002", "FD004")        # these two subsets have 6 flight conditions
CONDS = [(0, .0), (10, .25), (20, .7), (25, .62), (35, .84), (42, .84)]
COND_OFF = rng.normal(0, 8, (6, N_SENS)) * SCALE

def engine(unit, life):
    rows = []
    for c in range(1, life + 1):
        deg = (c / life) ** 2.2
        k = int(rng.integers(0, 6)) if MULTI else 0
        op1, op2 = CONDS[k] if MULTI else (0.0, 0.0)
        s = BASE + DIR * SCALE * np.array([4 * deg if i in ACTIVE else 0 for i in range(N_SENS)]) + rng.normal(0, 0.25 * SCALE)
        if MULTI:
            s = s + COND_OFF[k]
        rows.append([unit, c, op1, op2, 100.0] + list(np.round(s, 4)))
    return rows

def write(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(" ".join(str(x) for x in r) + " \n")

train, test, rul = [], [], []
for u in range(1, a.engines + 1):
    train += engine(u, int(rng.integers(130, 300)))
for u in range(1, a.engines + 1):
    life = int(rng.integers(130, 300))
    cut = int(rng.integers(40, life - 5))      # test engines stop before failure
    test += engine(u, life)[:cut]
    rul.append(life - cut)
write(os.path.join(a.out, f"train_{a.subset}.txt"), train)
write(os.path.join(a.out, f"test_{a.subset}.txt"), test)
with open(os.path.join(a.out, f"RUL_{a.subset}.txt"), "w") as f:
    f.write("\n".join(str(x) for x in rul) + "\n")
print(f"wrote simulated {a.subset} files to {a.out}/  ({a.engines} train + {a.engines} test engines)")
