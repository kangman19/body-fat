"""Stage 3: body fat from measurements, and measurements for a chosen body fat.

Trained on NHANES 1999-2006 adults (18+): DXA total body fat % (DXDTOPF) plus tape
measurements. Separate models for men and women. Plain least squares, so every number
can be inspected in models/nhanes_bf.json.

Forward   estimate_bf():  height, weight, age, waist, arm, calf  ->  body fat %
          (plus the build rule below, for men with a slim waist and a muscular build)
Inverse   targets_for_bf(): chosen body fat % -> target weight and girths for THIS person

How mesh measurements map onto NHANES measurements:
    waist_iliac -> BMXWAIST (same definition)
    bicep       -> BMXARMC  (relaxed mid upper arm)
    calf        -> BMXCALF  (maximal calf)
    thigh       -> BMXTHICR only as a ratio: NHANES measures mid-thigh, the mesh tape is
                   the ISO upper thigh, so thigh is never fed to the forward model.

Build rule (set by hand, NOT fitted: no dataset we have holds chest girth or muscle
together with DXA body fat). NHANES tape measurements cannot tell muscle from fat, so a
slim, muscular person reads high. The estimate is moved to an athletic range when
either
  - waist (hip bone) / height is slim AND the build is muscular (fitted muscle dial, or
    waist / chest), or
  - the waist / chest taper alone is strongly athletic. (Waist / hip is not used:
    ordinary women are already 0.70-0.75.)
                    men                             women
    slim            at or below 0.46, none from 0.50   at or below 0.45, none from 0.48
    athletic range  10-14%                          18-22%
    strong taper    waist / chest 0.75 (from 0.80)  waist / chest 0.70 (from 0.75)
    high floor      30% at 0.56, 42% at 0.68        38% at 0.56, 48% at 0.68
Numbers in BUILD_RULE. The NHANES-only value is always returned alongside ("bf_nhanes").
The tapers only count for a shape measured from photos: a body generated from numbers
has Anny's template shape, which says nothing about the person (ANSUR men's waist /
chest runs 0.80-0.97, so the template's ~0.81 would read as athletic).

The range shown is +/- BF_RANGE_HALF (4 points), wider than the tape model's own
cross-validated error (3.0 men, 3.4 women) because the build rule and the photo fit
add error of their own. Slider floors (BF_FLOOR) are 8% men / 18% women; below the
leanest NHANES adults (11.2% / 16.5%) the inverse extends the data's trend.

The inverse keeps all lean mass. A body fat slider changes fat only:
  1. Weight. Lean mass = weight x (1 - body fat) stays fixed, so the weight at the new
     body fat is lean / (1 - new body fat).
  2. Girths. NHANES gives the typical girth for a sex, height, age, lean mass and fat
     mass. Each girth is moved by typical(same lean, new fat) / typical(same lean, now),
     which keeps the person's own build. In the data the waist carries most of the
     change; arm, calf and thigh change by their fat layer only (about a third as much).
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
BF_FLOOR = {"male": 8.0, "female": 18.0}     # lowest body fat the slider allows
BF_RANGE_HALF = 4.0                          # body fat is shown as estimate +/- this
INVERSE_KEYS = ["waist", "arm", "calf", "thigh"]
LEG_KEYS = ["thigh", "calf"]   # predicted from the upper body when a photo does not show the legs
# Each pair is (no effect, full effect). Waist / height uses the NHANES (hip bone) waist;
# the tapers use the narrowest waist.
BUILD_RULE = {
    "male": {
        "slim_whtr": (0.50, 0.46),
        "athletic_bf": (10.0, 14.0),       # at waist / height athletic_whtr .. slim
        "athletic_whtr": 0.40,
        "muscle_dial": (0.50, 0.75),       # Anny muscle dial: average .. clearly muscular
        "waist_chest": (0.90, 0.80),       # counts as muscular with a slim waist
        "strong_waist_chest": (0.80, 0.75),  # athletic on its own
        "high_whtr": (0.53, 0.56, 0.68),   # floor fades in from 0.53, high_bf[0] at 0.56, high_bf[1] at 0.68
        "high_bf": (30.0, 42.0),
    },
    "female": {
        "slim_whtr": (0.48, 0.45),
        "athletic_bf": (18.0, 22.0),
        "athletic_whtr": 0.39,
        "muscle_dial": (0.50, 0.75),
        # Strict: the bust lowers waist / chest whatever the build (an average body reads
        # about 0.80). Waist / hip is not used: ordinary women are 0.70-0.75.
        "waist_chest": (0.78, 0.72),
        "strong_waist_chest": (0.75, 0.70),
        "high_whtr": (0.53, 0.56, 0.68),
        "high_bf": (38.0, 48.0),
    },
}


def _forward_features(height_cm, weight_kg, age, waist_cm, arm_cm, calf_cm):
    bmi = weight_kg / (height_cm / 100.0) ** 2
    return np.stack([np.ones_like(bmi), height_cm, weight_kg, age, waist_cm, arm_cm, calf_cm,
                     bmi, waist_cm / height_cm, 1.0 / bmi], axis=-1)


def _leg_features(height_cm, weight_kg, age, waist_cm, arm_cm):
    return np.stack([np.ones_like(arm_cm), height_cm, weight_kg, age, waist_cm, arm_cm], axis=-1)


def _inverse_features(lean_kg, fat_kg, height_cm, age):
    return np.stack([np.ones_like(fat_kg), lean_kg, fat_kg, fat_kg ** 2, height_cm, age,
                     lean_kg * fat_kg, np.sqrt(fat_kg)], axis=-1)


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
        fat = d["weight"].to_numpy() * y / 100
        Xi = _inverse_features(d["weight"].to_numpy() - fat, fat, d["height"].to_numpy(), d["age"].to_numpy())
        Xl = _leg_features(*(d[c].to_numpy() for c in ["height", "weight", "age", "waist", "arm"]))
        legs = {k: np.linalg.lstsq(Xl, d[k].to_numpy(), rcond=None)[0] for k in LEG_KEYS}
        model[sex] = {
            "n": int(len(d)),
            "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "forward": np.linalg.lstsq(X, y, rcond=None)[0].tolist(),
            # Girths (cm) from lean mass, fat mass, height and age, for the inverse.
            "inverse_lean_fat": {k: np.linalg.lstsq(Xi, d[k].to_numpy(), rcond=None)[0].tolist()
                                 for k in INVERSE_KEYS},
            # Mid-thigh and calf girth (cm) from height, weight, age, waist and upper arm:
            # how the legs of real people go with the rest of them.
            "legs": {k: c.tolist() for k, c in legs.items()},
            "legs_rmse_cm": {k: float(np.sqrt(np.mean((Xl @ c - d[k].to_numpy()) ** 2))) for k, c in legs.items()},
            # The lowest body fat any real adult in the data had (for reference; the
            # slider floor is BF_FLOOR).
            "bf_floor": float(np.ceil(y.min() * 10) / 10),
            "bf_percentiles_5_50_95": np.percentile(y, [5, 50, 95]).round(1).tolist(),
        }
    MODEL_PATH.parent.mkdir(exist_ok=True)
    MODEL_PATH.write_text(json.dumps(model, indent=1))
    return model


def load_model() -> dict:
    if MODEL_PATH.exists():
        model = json.loads(MODEL_PATH.read_text())
        if "bf_floor" in model["male"] and "legs" in model["male"] and "inverse_lean_fat" in model["male"]:
            return model
    return train()


def _forward(model, sex, age, height_cm, weight_kg, waist_mm, arm_mm, calf_mm):
    x = _forward_features(*(np.float64(v) for v in
                            (height_cm, weight_kg, age, waist_mm / 10, arm_mm / 10, calf_mm / 10)))
    return float(x @ np.array(model[sex]["forward"]))


def _ramp(x, lo, hi):
    """0 at lo, 1 at hi, straight in between (lo may be above hi)."""
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def build_rule(sex, height_cm, girths_mm, muscle_dial, bf_nhanes, measured_shape=True) -> float:
    """Body fat after the hand-set build rule (see BUILD_RULE and the module notes)."""
    r = BUILD_RULE[sex]
    whtr = girths_mm["waist_iliac"] / 10 / height_cm
    bf = bf_nhanes
    ratios = {}
    if measured_shape and girths_mm.get("chest"):
        ratios["waist_chest"] = girths_mm["waist"] / girths_mm["chest"]
    muscular = max([_ramp(x, *r[k]) for k, x in ratios.items()] +
                   ([_ramp(muscle_dial, *r["muscle_dial"])] if muscle_dial is not None else []) + [0.0])
    strong = max([_ramp(x, *r["strong_" + k]) for k, x in ratios.items()] + [0.0])
    share = max(_ramp(whtr, *r["slim_whtr"]) * muscular, strong)
    lo, hi = r["athletic_bf"]
    athletic = lo + (hi - lo) * _ramp(whtr, r["athletic_whtr"], r["slim_whtr"][1])
    bf = bf + share * min(athletic - bf, 0.0)          # only ever lowers the estimate
    fade, at, top = r["high_whtr"]
    floor = r["high_bf"][0] + (r["high_bf"][1] - r["high_bf"][0]) * _ramp(whtr, at, top)
    return bf + _ramp(whtr, fade, at) * max(floor - bf, 0.0)   # only ever raises it


def estimate_bf(model, sex, age, height_cm, weight_kg, girths_mm, muscle_dial=None,
                measured_shape=True) -> dict:
    """Body fat % as a midpoint plus a range of +/- BF_RANGE_HALF.
    muscle_dial: the body's fitted Anny muscle dial (None leaves the build rule's muscle
    test to the waist / chest taper alone). measured_shape: False for a body generated
    from numbers (its tapers are the template's, so the build rule ignores them)."""
    nhanes = _forward(model, sex, age, height_cm, weight_kg,
                      girths_mm["waist_iliac"], girths_mm["bicep"], girths_mm["calf"])
    mid = build_rule(sex, height_cm, girths_mm, muscle_dial, nhanes, measured_shape)
    half = BF_RANGE_HALF
    return {"bf_mid": round(mid, 1), "bf_range": [round(mid - half, 1), round(mid + half, 1)],
            "bf_nhanes": round(nhanes, 1)}


def legs_from_upper(model, sex, age, height_cm, weight_kg, waist_iliac_mm, bicep_mm) -> dict:
    """Typical mid-thigh and calf girth (mm) for someone with this height, weight, age,
    waist and upper arm. Used only for legs a photo does not show."""
    x = _leg_features(*(np.float64(v) for v in (height_cm, weight_kg, age, waist_iliac_mm / 10, bicep_mm / 10)))
    return {"thigh_mid": float(x @ np.array(model[sex]["legs"]["thigh"])) * 10,
            "calf": float(x @ np.array(model[sex]["legs"]["calf"])) * 10}


def clamp_target(model, sex, target_bf):
    return float(min(max(target_bf, BF_FLOOR[sex]), BF_CEILING[sex]))


def targets_for_bf(model, sex, age, height_cm, weight_kg, girths_mm, target_bf, bf_now) -> dict:
    """Target weight and girths for this person at target_bf (clamped to the safe range),
    with all lean mass kept. bf_now is the body's current estimate (estimate_bf).

    Returns {"target_bf", "mass_kg", "lean_kg", "girths_mm": {waist_iliac, bicep, calf,
    thigh, wrist, and forearm if girths_mm has it}}. Wrist is held at its current value:
    it is bone, not fat. NHANES has no forearm tape; it follows the upper arm.
    """
    target_bf = clamp_target(model, sex, target_bf)
    coef = {k: np.array(v) for k, v in model[sex]["inverse_lean_fat"].items()}
    trend = lambda lean, fat: {
        k: float(_inverse_features(*(np.float64(x) for x in (lean, fat, height_cm, age))) @ c)
        for k, c in coef.items()}
    # Step 1: lean mass stays, so the weight follows from the target body fat.
    lean = weight_kg * (1 - bf_now / 100)
    mass = lean / (1 - target_bf / 100)
    # Step 2: girths of real people with the same lean mass, now and at the target.
    base = trend(lean, weight_kg - lean)
    r = {k: v / base[k] for k, v in trend(lean, mass - lean).items()}
    return {"target_bf": target_bf, "mass_kg": mass, "lean_kg": lean,
            "girths_mm": {"waist_iliac": girths_mm["waist_iliac"] * r["waist"],
                          "bicep": girths_mm["bicep"] * r["arm"],
                          "calf": girths_mm["calf"] * r["calf"],
                          "thigh": girths_mm["thigh"] * r["thigh"],
                          "wrist": girths_mm["wrist"],
                          **({"forearm": girths_mm["forearm"] * r["arm"]} if "forearm" in girths_mm else {})}}


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
              "% points   leanest in data:", m[sex]["bf_floor"], " slider floor:", BF_FLOOR[sex],
              "  5th/50th/95th percentile:", m[sex]["bf_percentiles_5_50_95"])
    g = {"waist_iliac": 932, "bicep": 354, "calf": 398, "thigh": 598, "wrist": 176}
    print("example male 178 cm, 78 kg:", estimate_bf(m, "male", 30, 178, 78, g))
    now = estimate_bf(m, "male", 30, 178, 78, g)["bf_mid"]
    for bf in (5, 15, 25, 35):
        t = targets_for_bf(m, "male", 30, 178, 78, g, bf, now)
        print(f"  target {bf}% -> {t['mass_kg']:.1f} kg", {k: round(v) for k, v in t["girths_mm"].items()})
