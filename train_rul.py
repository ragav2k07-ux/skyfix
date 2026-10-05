#!/usr/bin/env python3
"""SkyFix - train a Remaining Useful Life (RUL) model on NASA C-MAPSS and write predictions.csv.

Usage:
    python train_rul.py --data CMAPSSData --subset FD001
    python train_rul.py --data CMAPSSData --subset FD004 --out predictions.csv

Expects train_FDxxx.txt, test_FDxxx.txt, RUL_FDxxx.txt in --data (NASA C-MAPSS).
Output columns (what the SkyFix website reads):
    unit, aircraft, cycle, pred_rul, pred_lo, pred_hi, true_rul, <sensor columns>
"""
import argparse, os, sys
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

CLIP = 125          # piece-wise linear RUL cap (standard C-MAPSS practice)
WIN = 5             # smoothing window (cycles)
COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]


def load(data, subset):
    rd = lambda n: pd.read_csv(os.path.join(data, f"{n}_{subset}.txt"), sep=r"\s+", header=None)
    tr, te = rd("train"), rd("test")
    tr, te = tr.iloc[:, :26], te.iloc[:, :26]
    tr.columns = te.columns = COLS
    rul = pd.read_csv(os.path.join(data, f"RUL_{subset}.txt"), header=None)[0].values
    return tr, te, rul


def add_true_rul(tr, te, rul):
    tr["true_rul"] = tr.groupby("unit")["cycle"].transform("max") - tr["cycle"]
    last = te.groupby("unit")["cycle"].transform("max")
    te["true_rul"] = te["unit"].map(dict(zip(sorted(te.unit.unique()), rul))) + (last - te["cycle"])
    return tr, te


def normalise(tr, te, sensors, n_cond):
    """Per-operating-condition z-score so 0 means 'like a healthy new engine'."""
    ops = ["op1", "op2", "op3"]
    if n_cond > 1:                       # FD002 / FD004 have 6 flight conditions
        km = KMeans(n_clusters=n_cond, n_init=10, random_state=0).fit(tr[ops])
        tr["cond"], te["cond"] = km.predict(tr[ops]), km.predict(te[ops])
    else:
        tr["cond"] = te["cond"] = 0
    for c in tr["cond"].unique():
        m = tr["cond"] == c
        mu, sd = tr.loc[m, sensors].mean(), tr.loc[m, sensors].std().replace(0, 1)
        tr.loc[m, sensors] = (tr.loc[m, sensors] - mu) / sd
        mt = te["cond"] == c
        te.loc[mt, sensors] = (te.loc[mt, sensors] - mu) / sd
    return tr, te


def features(df, sensors):
    """Smoothed value, 10-cycle trend and drift from the engine's own start."""
    g = df.groupby("unit")
    out = df[["unit", "cycle"]].copy()
    for s in sensors:
        sm = g[s].transform(lambda x: x.rolling(WIN, min_periods=1).mean())
        out[s] = sm
        out[s + "_trend"] = sm - g[s].transform(lambda x: x.rolling(WIN, min_periods=1).mean().shift(10)).fillna(sm)
        out[s + "_drift"] = sm - g[s].transform(lambda x: x.iloc[:5].mean())
    out["cycle_n"] = df["cycle"]
    return out.drop(columns=["unit", "cycle"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="CMAPSSData")
    ap.add_argument("--subset", default="FD001", choices=["FD001", "FD002", "FD003", "FD004"])
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--engines", type=int, default=0, help="limit engines written (0 = all)")
    a = ap.parse_args()

    tr, te, rul = load(a.data, a.subset)
    tr, te = add_true_rul(tr, te, rul)
    n_cond = 6 if a.subset in ("FD002", "FD004") else 1
    sensors = [s for s in COLS[5:] if tr[s].std() > 1e-6]      # drop flat sensors
    tr, te = normalise(tr, te, sensors, n_cond)

    Xtr, Xte = features(tr, sensors), features(te, sensors)
    ytr = tr["true_rul"].clip(upper=CLIP)

    # hold out 20% of training engines for honest validation + calibrated range
    units = tr["unit"].unique(); rng = np.random.default_rng(0); rng.shuffle(units)
    val_u = set(units[: max(1, len(units) // 5)]); is_val = tr["unit"].isin(val_u)
    mk = lambda **k: HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, max_leaf_nodes=24, random_state=0, **k)
    mid = mk().fit(Xtr[~is_val], ytr[~is_val])
    lo = mk(loss="quantile", quantile=0.10).fit(Xtr[~is_val], ytr[~is_val])
    hi = mk(loss="quantile", quantile=0.90).fit(Xtr[~is_val], ytr[~is_val])
    rmse_val = mean_squared_error(ytr[is_val], mid.predict(Xtr[is_val])) ** 0.5

    # final refit on all training engines
    mid, lo, hi = mk().fit(Xtr, ytr), mk(loss="quantile", quantile=0.10).fit(Xtr, ytr), mk(loss="quantile", quantile=0.90).fit(Xtr, ytr)
    p = np.clip(mid.predict(Xte), 0, CLIP)
    pl, ph = np.clip(lo.predict(Xte), 0, CLIP), np.clip(hi.predict(Xte), 0, CLIP)
    pl, ph = np.minimum(pl, p), np.maximum(ph, p)

    # test-set score: last cycle of every engine (the official C-MAPSS metric)
    last = te.groupby("unit")["cycle"].transform("max") == te["cycle"]
    err = p[last.values] - te.loc[last, "true_rul"].clip(upper=CLIP).values
    rmse = float(np.sqrt(np.mean(err ** 2)))
    d = p[last.values] - te.loc[last, "true_rul"].values
    nasa = float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))
    cover = float(np.mean((te.loc[last, "true_rul"].clip(upper=CLIP).values >= pl[last.values]) &
                          (te.loc[last, "true_rul"].clip(upper=CLIP).values <= ph[last.values])))
    print(f"{a.subset}: validation RMSE {rmse_val:.1f} | TEST RMSE {rmse:.1f} cycles | NASA score {nasa:.0f} | 80% range covers {cover:.0%}")

    res = pd.DataFrame({"unit": te["unit"], "aircraft": (te["unit"] + 1) // 2, "cycle": te["cycle"],
                        "pred_rul": p.round(1), "pred_lo": pl.round(1), "pred_hi": ph.round(1),
                        "true_rul": te["true_rul"].clip(upper=CLIP)})
    for s in sensors[:9]:
        res[s] = te[s].round(3)
    if a.engines:
        res = res[res["unit"].isin(sorted(res.unit.unique())[: a.engines])]
    res.to_csv(a.out, index=False)
    print(f"wrote {a.out}  ({res.unit.nunique()} engines, {len(res)} rows)")


if __name__ == "__main__":
    sys.exit(main())
