"""Stage 3: body fat from measurements, and measurements for a chosen body fat.

Trained on NHANES 1999-2006 adults (18+): DXA total body fat % (DXDTOPF) plus tape
measurements. Separate models for men and women. Plain least squares, so every number
can be inspected in models/nhanes_bf.json.

Forward   estimate_bf():  height, weight, age, waist, arm, calf  ->  body fat %
Inverse   targets_for_bf(): chosen body fat % -> target weight and girths for THIS person

How mesh measurements map onto NHANES measurements:
    waist_iliac -> BMXWAIST (same definition)
    bicep       -> BMXARMC  (relaxed mid upper arm)
    calf        -> BMXCALF  (maximal calf)
    thigh       -> BMXTHICR only as a ratio: NHANES measures mid-thigh, the mesh tape is
                   the ISO upper thigh, so thigh is never fed to the forward model.

The inverse works in two steps:
  1. Weight. Fat mass changes; lean mass changes by LEAN_FRACTION of the weight change
     (the common "a quarter of weight change is lean tissue" rule of thumb). That fixes
     the weight at the new body fat.
  2. Girths. NHANES gives the typical girth for a sex, height, age, weight and body fat.
     Each girth is moved by the ratio typical(new) / typical(current), which keeps the
     person's own build.
The targets are NOT forced to make the forward model return the requested body fat:
doing so gave unrealistically small waists at low body fat. Checked against real NHANES
men under 13% fat (median 62.5 kg, waist 716 mm, arm 286 mm, calf 348 mm), the plain
trend lands close (59.9 kg, 713, 296, 353 for a 10% target from a 178 cm, 78 kg start).
It is a population trend, not a prediction of how one particular person would change.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NHANES = ROOT / "data" / "nhanes"
MODEL_PATH = ROOT / "models" / "nhanes_bf.json"

BF_CEILING = {"male": 45.0, "female": 55.0}
INVERSE_KEYS = ["waist", "arm", "calf", "thigh"]
LEAN_FRACTION = 0.25  # share of any weight change that is lean tissue


def _forward_features(height_cm, weight_kg, age, waist_cm, arm_cm, calf_cm):
    bmi = weight_kg / (height_cm / 100.0) ** 2
    return np.stack([np.ones_like(bmi), height_cm, weight_kg, age, waist_cm, arm_cm, calf_cm,
                     bmi, waist_cm / height_cm, 1.0 / bmi], axis=-1)


def _inverse_features(bf, weight_kg, height_cm, age):
    return np.stack([np.ones_like(bf), bf, weight_kg, height_cm, age, bf * weight_kg], axis=-1)


def load_nhanes() -> pd.DataFrame:
    frames = []
    for suffix in ["", "_B", "_C", "_D"]:
        demo = pd.read_sas(NHANES / f"DEMO{suffix}.XPT")[["SEQN", "RIAGENDR", "RIDAGEYR"]]
        bmx = pd.read_sas(NHANES / f"BMX{suffix}.XPT")[
            ["SEQN", "BMXWT", "BMXHT", "BMXWAIST", "BMXARMC", "BMXCALF", "BMXTHICR"]]
        # DXA has 5 imputed rows per person; average them into one body fat value.
        dxx = pd.read_sas(NHANES / f"DXX{suffix}.XPT")[["SEQN", "DXDTOPF"]]
        dxx = dxx.groupby("SEQN", as_index=False)["DXDTOPF"].mean()
        frames.append(demo.merge(bmx, on="SEQN").merge(dxx, on="SEQN"))
    df = pd.concat(frames, ignore_index=True).dropna()
    df = df[df["RIDAGEYR"] >= 18]
    df["sex"] = np.where(df["RIAGENDR"] == 1, "male", "female")
    return df.rename(columns={
        "RIDAGEYR": "age", "BMXWT": "weight", "BMXHT": "height", "BMXWAIST": "waist",
        "BMXARMC": "arm", "BMXCALF": "calf", "BMXTHICR": "thigh", "DXDTOPF": "bf"})


def train() -> dict:
    df = load_nhanes()
    model = {"source": "NHANES 1999-2006, adults 18+, DXA total percent fat"}
    rng = np.random.default_rng(0)
    for sex in ("male", "female"):
        d = df[df["sex"] == sex]
        X = _forward_features(*(d[c].to_numpy() for c in ["height", "weight", "age", "waist", "arm", "calf"]))
        y = d["bf"].to_numpy()
        # 5-fold cross-validation: error on people the model has not seen.
        fold = rng.integers(0, 5, len(d))
        errors = np.concatenate([
            X[fold == k] @ np.linalg.lstsq(X[fold != k], y[fold != k], rcond=None)[0] - y[fold == k]
            for k in range(5)])
        Xi = _inverse_features(y, d["weight"].to_numpy(), d["height"].to_numpy(), d["age"].to_numpy())
        model[sex] = {
            "n": int(len(d)),
            "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "forward": np.linalg.lstsq(X, y, rcond=None)[0].tolist(),
            "inverse": {k: np.linalg.lstsq(Xi, d[k].to_numpy(), rcond=None)[0].tolist()
                        for k in INVERSE_KEYS},
            # Slider floor: the lowest body fat any real adult in the data had, so the
            # inverse never goes below what was actually measured.
            "bf_floor": float(np.ceil(y.min() * 10) / 10),
            "bf_percentiles_5_50_95": np.percentile(y, [5, 50, 95]).round(1).tolist(),
        }
    MODEL_PATH.parent.mkdir(exist_ok=True)
    MODEL_PATH.write_text(json.dumps(model, indent=1))
    return model


def load_model() -> dict:
    if MODEL_PATH.exists():
        model = json.loads(MODEL_PATH.read_text())
        if "bf_floor" in model["male"]:
            return model
    return train()


def _forward(model, sex, age, height_cm, weight_kg, waist_mm, arm_mm, calf_mm):
    x = _forward_features(*(np.float64(v) for v in
                            (height_cm, weight_kg, age, waist_mm / 10, arm_mm / 10, calf_mm / 10)))
    return float(x @ np.array(model[sex]["forward"]))


def estimate_bf(model, sex, age, height_cm, weight_kg, girths_mm) -> dict:
    """Body fat % as a midpoint plus a range of +/- one cross-validated RMSE."""
    mid = _forward(model, sex, age, height_cm, weight_kg,
                   girths_mm["waist_iliac"], girths_mm["bicep"], girths_mm["calf"])
    half = model[sex]["rmse"]
    return {"bf_mid": round(mid, 1), "bf_range": [round(mid - half, 1), round(mid + half, 1)]}


def clamp_target(model, sex, target_bf):
    return float(min(max(target_bf, model[sex]["bf_floor"]), BF_CEILING[sex]))


def targets_for_bf(model, sex, age, height_cm, weight_kg, girths_mm, target_bf) -> dict:
    """Target weight and girths for this person at target_bf (clamped to the safe range).

    Returns {"target_bf", "mass_kg", "girths_mm": {waist_iliac, bicep, calf, thigh, wrist}}.
    Wrist is held at its current value: it is bone, not fat.
    """
    target_bf = clamp_target(model, sex, target_bf)
    coef = {k: np.array(v) for k, v in model[sex]["inverse"].items()}
    trend = lambda bf, kg: {
        k: float(_inverse_features(*(np.float64(x) for x in (bf, kg, height_cm, age))) @ c)
        for k, c in coef.items()}
    bf_now = _forward(model, sex, age, height_cm, weight_kg,
                      girths_mm["waist_iliac"], girths_mm["bicep"], girths_mm["calf"])
    # Step 1: weight at the target body fat, with lean mass following LEAN_FRACTION.
    mass = weight_kg * (1 - bf_now / 100 - LEAN_FRACTION) / (1 - target_bf / 100 - LEAN_FRACTION)
    base = trend(bf_now, weight_kg)

    r = {k: v / base[k] for k, v in trend(target_bf, mass).items()}
    return {"target_bf": target_bf, "mass_kg": mass,
            "girths_mm": {"waist_iliac": girths_mm["waist_iliac"] * r["waist"],
                          "bicep": girths_mm["bicep"] * r["arm"],
                          "calf": girths_mm["calf"] * r["calf"],
                          "thigh": girths_mm["thigh"] * r["thigh"],
                          "wrist": girths_mm["wrist"]}}


def write_bf_json(path, estimate, targets):
    """bf.json contract (stage 3 output)."""
    Path(path).write_text(json.dumps({
        **estimate, "target_bf": targets["target_bf"],
        "target_girths_mm": {k: round(v, 1) for k, v in targets["girths_mm"].items()},
    }, indent=2))


if __name__ == "__main__":
    m = train()
    for sex in ("male", "female"):
        print(sex, "n =", m[sex]["n"], " cross-validated RMSE =", round(m[sex]["rmse"], 2),
              "% points   floor:", m[sex]["bf_floor"],
              "  5th/50th/95th percentile:", m[sex]["bf_percentiles_5_50_95"])
    g = {"waist_iliac": 932, "bicep": 354, "calf": 398, "thigh": 598, "wrist": 176}
    print("example male 178 cm, 78 kg:", estimate_bf(m, "male", 30, 178, 78, g))
    for bf in (5, 15, 25, 35):
        t = targets_for_bf(m, "male", 30, 178, 78, g, bf)
        print(f"  target {bf}% -> {t['mass_kg']:.1f} kg", {k: round(v) for k, v in t["girths_mm"].items()})
