"""Stage 4: inverse morph. Given target girths, find the Anny shape that has them.

Height is locked: AnnyBody.build() rescales every candidate mesh to exactly height_cm,
so the optimiser can never trade height for girth. Sex and age are fixed too.

What Adam is allowed to move:
    dials  weight, muscle, proportions, height  (whole-body shape; the height dial only
           changes build, since the final height is locked)
    local  SHAPE_LOCALS: per-region girth / width / depth / fat sliders, pulled towards
           zero (or towards their starting values) so the whole-body dials do most of
           the work

What is set, not solved, in a body fat morph (fixed_for_bf, FRAME_DIALS):
    definition  DEFINITION_LOCALS: abdominal tautness, straight from body fat.
    muscle      MUSCLE_LOCALS and the muscle dial: the person's own muscle, unchanged.
                Losing or gaining fat never grows or melts muscle.
    frame       the proportions and height dials and the shoulder width slider: bone,
                which body fat does not change.
    The solver works around them, so the girth targets are still met with them applied.

The finer muscle contours (abdominal blocks, serratus, deltoid and quadriceps
separations) are too small for Anny's mesh; they are surface relief, see stage 5.

Targets may also include "mass_kg" (weight from mesh volume).

Loss = mean squared relative target error
     + local_weight * mean((local - anchor)^2)
     + a term keeping Anny's natural height close to height_cm (so rescaling stays ~1).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import trimesh

from stage2_measure import (DEFINITION_LOCALS, DIAL_RANGES, EXTRA_GIRTH_KEYS, LOCAL_LABELS,
                            MUSCLE_LOCALS, AnnyBody, default_dials)

# Body fat (DXA %) at or below which definition is full, and at or above which the
# surface is completely smooth.
DEFINITION_BF = {"male": (8.0, 25.0), "female": (18.0, 32.0)}
DEFINITION_STRENGTH = {"male": 1.0, "female": 0.6}   # women carry less visible relief at the same leanness
FRAME_DIALS = ["muscle", "proportions", "height"]     # held at the person's values in a body fat morph
FRAME_LOCALS = ["measure-shoulder-dist-incr"]          # same, among the shape sliders
# Waist is the measurement body fat depends on most, so missing it costs more.
TARGET_WEIGHT = {"waist": 4.0, "waist_iliac": 4.0, "waist_navel": 4.0}


def definition(sex: str, bf: float) -> float:
    """0 = smooth, 1 = fully defined."""
    lo, hi = DEFINITION_BF[sex]
    return float(min(max((hi - bf) / (hi - lo), 0.0), 1.0))


def relief(sex: str, bf: float) -> float:
    """Strength of visible muscle relief, 0..1, for this sex and body fat."""
    return DEFINITION_STRENGTH[sex] * definition(sex, bf)


def fixed_for_bf(sex: str, target_bf: float, start_local: dict | None = None) -> dict[str, float]:
    """Sliders that are set, not solved, for a body at target_bf: abdominal tautness from
    body fat, and the person's own muscle and shoulder width as they are. start_local
    describes the body being morphed (leave empty when there is none)."""
    start_local = start_local or {}
    fixed = {k: relief(sex, target_bf) for k in DEFINITION_LOCALS}
    fixed.update({k: float(start_local.get(k, 0.0)) for k in MUSCLE_LOCALS + FRAME_LOCALS})
    return fixed


def _to_raw(value, lo, hi):
    x = min(max((value - lo) / (hi - lo), 1e-4), 1 - 1e-4)
    return float(np.log(x / (1 - x)))


def solve(
    anny_body: AnnyBody,
    sex: str,
    age: float,
    height_cm: float,
    targets: dict[str, float],
    init_dials: dict[str, float] | None = None,
    init_local: dict[str, float] | None = None,
    fixed_local: dict[str, float] | None = None,
    fixed_dials: list[str] | tuple = (),
    anchor_local: bool = False,
    steps: int = 200,
    lr: float = 0.05,
    local_weight: float = 0.02,
    relocate_every: int = 10,
    log=None,
):
    """Returns (dials, local, body, girths_mm, history).

    targets       girths in mm by name, optionally "mass_kg".
    fixed_local   sliders applied but not optimised (see fixed_for_bf).
    fixed_dials   dials kept at their init_dials values (see FRAME_DIALS).
    anchor_local  pull local sliders towards init_local instead of towards zero. Use it
                  when morphing a fitted person, so their own build is kept.

    `local` in the result holds the solved sliders plus fixed_local, so build() with it
    reproduces the body. Each dial is stored unbounded and squashed into its range with
    a sigmoid; local sliders are squashed into [-1, 1] with tanh.
    """
    init_dials = {**default_dials(), **(init_dials or {})}
    init_local = {k: float(np.clip((init_local or {}).get(k, 0.0), -0.999, 0.999)) for k in LOCAL_LABELS}
    fixed_local = dict(fixed_local or {})
    raw_d = {
        k: torch.tensor(_to_raw(init_dials[k], *DIAL_RANGES[k]), requires_grad=k not in fixed_dials)
        for k in DIAL_RANGES
    }
    raw_l = {k: torch.tensor(float(np.arctanh(init_local[k])), requires_grad=True) for k in LOCAL_LABELS}
    anchor = init_local if anchor_local else {k: 0.0 for k in LOCAL_LABELS}

    def current():
        dials = {k: DIAL_RANGES[k][0] + (DIAL_RANGES[k][1] - DIAL_RANGES[k][0]) * torch.sigmoid(r)
                 for k, r in raw_d.items()}
        local = {k: torch.tanh(r) for k, r in raw_l.items()}
        return dials, local

    opt = torch.optim.Adam([r for r in raw_d.values() if r.requires_grad] + list(raw_l.values()), lr=lr)
    history, landmarks = [], None
    full = any(k in EXTRA_GIRTH_KEYS for k in targets)   # a target among the extra tapes
    for step in range(steps):
        opt.zero_grad()
        dials, local = current()
        body = anny_body.build(sex, age, height_cm, dials, {**local, **fixed_local})
        if step % relocate_every == 0:  # tape positions move slowly; refresh now and then
            landmarks = anny_body.landmarks(body, full=full)
        measured = anny_body.measure(body, landmarks, full=full)
        measured["mass_kg"] = body["mass_kg"]
        loss = torch.stack([TARGET_WEIGHT.get(k, 1.0) * ((measured[k] - t) / t) ** 2
                            for k, t in targets.items()]).mean()
        loss = loss + local_weight * torch.stack([(local[k] - anchor[k]) ** 2 for k in LOCAL_LABELS]).mean()
        loss = loss + 10.0 * (body["natural_height_m"] - height_cm / 100.0) ** 2
        loss.backward()
        opt.step()
        worst = max(abs(float(measured[k].detach()) - t) for k, t in targets.items() if k != "mass_kg")
        history.append({"step": step, "loss": float(loss), "worst_girth_error_mm": worst})
        if log and (step % 25 == 0 or step == steps - 1):
            log(f"step {step:3d}  loss {float(loss):.6f}  worst girth error {worst:6.1f} mm")

    with torch.no_grad():
        dials, local = current()
        dials = {k: float(v) for k, v in dials.items()}
        local = {**{k: float(v) for k, v in local.items()}, **fixed_local}
        body = anny_body.build(sex, age, height_cm, dials, local)
        girths = {k: float(v) for k, v in anny_body.measure(body).items()}
    return dials, local, body, girths, history


def export(anny_body: AnnyBody, body, dials, local, out_dir, name="morphed_mesh"):
    """Write <name>.glb (Y-up, facing +Z, for Three.js) and morph_params.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    v = body["vertices"].detach().numpy()
    v_yup = np.stack([v[:, 0], v[:, 2], -v[:, 1]], axis=1)
    mesh = trimesh.Trimesh(v_yup, anny_body.faces.numpy(), process=False)
    mesh.export(out_dir / f"{name}.glb")
    (out_dir / "morph_params.json").write_text(
        json.dumps({"dials": dials, "local": local}, indent=2)
    )
    return out_dir / f"{name}.glb"
