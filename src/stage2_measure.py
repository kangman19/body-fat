"""Stage 2: canonical Anny body + virtual tape measure.

Two things live here:

1. AnnyBody   - builds a standing A-pose Anny mesh, always rescaled so its height is
                exactly the user's height_cm (the scale anchor), feet on z = 0.
2. measure()  - slices that mesh with planes and returns convex-hull girths in mm.
                A convex hull behaves like a tape pulled tight: it bridges hollows
                instead of dipping into them.

Axes (Anny rest pose): x = body's left, y = backwards (the face looks along -y), z = up.

Everything is written in torch so stage4_morph.py can back-propagate from a girth to
the Anny shape parameters.

Waist is measured three ways and never converted between them:
    waist        ISO 8559-1 natural waist: greatest indentation of the torso
    waist_iliac  NHANES: just above the iliac crest
    waist_navel  ANSUR II "waistcircumference": at the navel (omphalion)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("ANNY_CACHE_DIR", str(ROOT / ".anny_cache"))

import numpy as np
import roma
import torch
from scipy.spatial import ConvexHull

import anny

DENSITY_KG_M3 = 980.0  # same whole-body density Anny's own anthropometry assumes
GENDER_DIAL = {"male": 0.0, "female": 1.0}  # Anny: gender anchors are [male, female]

# Anny has no pelvis bone surface, so the iliac crest height comes from the ANSUR II
# mean ratio iliocristale height / stature (computed from data/ansur).
ILIAC_CREST_RATIO = {"male": 0.6044, "female": 0.6113}
ABOVE_CREST_M = 0.01  # NHANES tape sits just above the crest

# Shape dials the solvers are allowed to move, with the range each may take.
# Weight goes far past Anny's nominal 0..1 because 1.0 is only about BMI 22.
DIAL_RANGES = {
    "weight": (-0.75, 4.0),
    "muscle": (0.0, 1.0),
    "proportions": (0.0, 1.0),
    "height": (0.0, 1.0),
}
# Anny local sliders, in three groups. A name without "l-"/"r-" in front drives both sides.
# SHAPE_LOCALS: fat, width, depth and girth. Moved by every solver.
SHAPE_LOCALS = [
    "measure-waist-circ-incr", "measure-hips-circ-incr", "measure-thigh-circ-incr",
    "measure-calf-circ-incr", "measure-upperarm-circ-incr", "measure-bust-circ-incr",
    "measure-underbust-circ-incr", "measure-wrist-circ-incr", "measure-knee-circ-incr",
    "measure-ankle-circ-incr", "measure-shoulder-dist-incr", "measure-neck-circ-incr",
    "neck-scale-horiz-incr",
    "torso-scale-horiz-incr", "torso-scale-depth-incr",
    "hip-scale-horiz-incr", "hip-scale-depth-incr", "buttocks-volume-incr",
    "upperarm-fat-incr", "lowerarm-fat-incr", "upperleg-fat-incr", "lowerleg-fat-incr",
    "upperarm-scale-horiz-incr", "upperarm-scale-depth-incr",
    "lowerarm-scale-horiz-incr", "lowerarm-scale-depth-incr",
    "upperleg-scale-horiz-incr", "upperleg-scale-depth-incr",
    "lowerleg-scale-horiz-incr", "lowerleg-scale-depth-incr",
]
# MUSCLE_LOCALS: muscle volume by region. Fitted to the person's photo outline (stage 1).
# In a morph they are not solved: they stay at the person's values, shifted a little
# with body fat (stage4_morph.muscle_locals), so losing fat does not grow or melt muscle.
MUSCLE_LOCALS = [
    "torso-muscle-pectoral-incr", "torso-muscle-dorsi-incr", "torso-vshape-incr",
    "upperarm-shoulder-muscle-incr", "upperarm-muscle-incr", "lowerarm-muscle-incr",
    "upperleg-muscle-incr", "lowerleg-muscle-incr",
]
# DEFINITION_LOCALS: how taut the abdomen is. Set from body fat only, never fitted.
DEFINITION_LOCALS = ["stomach-tone-incr", "pelvis-tone-incr"]
LOCAL_LABELS = SHAPE_LOCALS

GIRTH_KEYS = [
    "waist", "waist_iliac", "waist_navel",
    "hip", "thigh", "bicep", "calf", "chest", "wrist",
]


def age_to_dial(age_years: float) -> float:
    """Anny's age dial: 2/3 is 'young adult', 1.0 is 'old'. Linear from 25 to 90 years."""
    t = (float(age_years) - 25.0) / (90.0 - 25.0)
    return 2.0 / 3.0 + min(max(t, 0.0), 1.0) / 3.0


def default_dials() -> dict[str, float]:
    return {"weight": 0.5, "muscle": 0.5, "proportions": 0.5, "height": 0.5}


class AnnyBody:
    """Anny in its rest A-pose, scaled to an exact height."""

    def __init__(self):
        self.model = anny.Anny(
            local_changes="default", extrapolate_phenotypes=True
        ).to(dtype=torch.float32)
        m = self.model
        self.faces = m.faces.long()
        self.bone = {label: i for i, label in enumerate(m.bone_labels)}

        # Dominant bone per vertex -> which vertices belong to an arm (incl. hand).
        top = m.vertex_bone_weights.argmax(dim=1, keepdim=True)
        dominant = m.vertex_bone_indices.gather(1, top).squeeze(1)
        arm_words = ("upperarm", "lowerarm", "wrist", "finger", "metacarpal")
        is_arm = torch.tensor([any(w in l for w in arm_words) for l in m.bone_labels])
        is_left = torch.tensor([l.endswith(".L") for l in m.bone_labels])
        v_arm = is_arm[dominant]
        v_left_arm = v_arm & is_left[dominant]

        # Vertices used when comparing surfaces with another body model: everything but
        # head, hands and feet, where the two models differ for reasons unrelated to build.
        skip_words = ("head", "eye", "wrist", "finger", "metacarpal", "foot", "toe")
        is_skip = torch.tensor([any(w in l for w in skip_words) for l in m.bone_labels])
        self.fit_mask = ~is_skip[dominant]
        # Body part of every vertex, for anything that treats regions differently.
        def part(label):
            for words, name in ((("finger", "metacarpal", "wrist"), "hand"), (("toe", "foot"), "foot"),
                                (("head", "eye"), "head"), (("neck",), "neck"), (("upperarm",), "upperarm"),
                                (("lowerarm",), "lowerarm"), (("upperleg",), "upperleg"),
                                (("lowerleg",), "lowerleg")):
                if any(w in label for w in words):
                    return name
            return "torso"
        self.vertex_part = np.array([part(m.bone_labels[int(i)]) for i in dominant])
        self._local_labels = set(m.local_change_labels)

        f = self.faces
        self.edges_trunk = _unique_edges(f[~v_arm[f].any(dim=1)])  # torso + legs + head
        self.edges_left_arm = _unique_edges(f[v_left_arm[f].all(dim=1)])

        # Navel vertex = the vertex Anny's own "navel out" slider moves the most.
        with torch.no_grad():
            base = m(phenotype_kwargs=self._phenotype("male", 30, default_dials()))
            out = m(
                phenotype_kwargs=self._phenotype("male", 30, default_dials()),
                local_changes_kwargs={"stomach-navel-out": 1.0},
            )
            moved = (out["rest_vertices"] - base["rest_vertices"])[0].norm(dim=1)
        self.navel_vertex = int(moved.argmax())

    def _phenotype(self, sex, age, dials):
        p = {"gender": GENDER_DIAL[sex], "age": age_to_dial(age)}
        p.update(dials)
        return {k: torch.as_tensor(v, dtype=torch.float32).reshape(1) for k, v in p.items()}

    def build(self, sex, age, height_cm, dials, local=None, pose=None):
        """Return a dict: vertices (V,3) m, joints {bone: (3,)}, natural_height_m, mass_kg.

        dials / local hold floats or torch scalars (gradients flow through them).
        The mesh is uniformly rescaled so max z - min z == height_cm exactly.

        pose: optional {bone label: rotation vector (3,)} to bend joints away from the
        rest A-pose. Adds "posed_vertices" and "posed_joints" (same scale, but in Anny's
        posed frame, whose origin differs from the rest frame). Measurements always use
        the rest pose.
        """
        expanded = {}
        for k, v in (local or {}).items():
            value = torch.as_tensor(v, dtype=torch.float32).reshape(1)
            for label in ([k] if k in self._local_labels else ["l-" + k, "r-" + k]):
                expanded[label] = value
        pose_matrices = None
        if pose is not None:
            pose_matrices = torch.eye(4).repeat(1, self.model.bone_count, 1, 1)
            for label, rotvec in pose.items():
                pose_matrices[0, self.bone[label], :3, :3] = roma.rotvec_to_rotmat(rotvec)
        out = self.model(
            pose_parameters=pose_matrices,
            phenotype_kwargs=self._phenotype(sex, age, dials),
            local_changes_kwargs=expanded or None,
        )
        v = out["rest_vertices"][0]
        heads = out["rest_bone_heads"][0]
        zmin = v[:, 2].min()
        natural_h = v[:, 2].max() - zmin
        scale = (height_cm / 100.0) / natural_h
        shift = torch.stack([torch.zeros(()), torch.zeros(()), zmin])
        v = (v - shift) * scale
        heads = (heads - shift) * scale
        body = {
            "vertices": v,
            "joints": {label: heads[i] for label, i in self.bone.items()},
            "natural_height_m": natural_h,
            "mass_kg": _volume(v, self.faces) * DENSITY_KG_M3,
            "sex": sex,
        }
        if pose is not None:
            body["posed_vertices"] = out["vertices"][0] * scale
            posed = out["bone_poses"][0, :, :3, 3] * scale
            body["posed_joints"] = {label: posed[i] for label, i in self.bone.items()}
        return body

    def from_height_weight_sex(self, height_cm, weight_kg, sex, age=30):
        """Generic body for a height/weight/sex: average muscle and proportions, Anny's
        height dial set so little rescaling is needed, weight dial solved by bisection
        so the mesh volume matches weight_kg. Returns (dials, body)."""
        dials = default_dials()
        with torch.no_grad():
            lo, hi = DIAL_RANGES["height"]
            for _ in range(25):  # natural height rises with the height dial
                dials["height"] = (lo + hi) / 2
                if float(self.build(sex, age, height_cm, dials)["natural_height_m"]) * 100 < height_cm:
                    lo = dials["height"]
                else:
                    hi = dials["height"]
            lo, hi = DIAL_RANGES["weight"]
            for _ in range(25):  # mass rises with the weight dial
                dials["weight"] = (lo + hi) / 2
                if float(self.build(sex, age, height_cm, dials)["mass_kg"]) < weight_kg:
                    lo = dials["weight"]
                else:
                    hi = dials["weight"]
            body = self.build(sex, age, height_cm, dials)
        return dials, body

    # ------------------------------------------------------------------ measuring

    def landmarks(self, body):
        """Find where each tape goes. No gradients: positions are treated as fixed
        while the girth at that position stays differentiable."""
        with torch.no_grad():
            v, j = body["vertices"], body["joints"]
            H = float(v[:, 2].max())
            z = lambda name: float(j[name][2])
            up = torch.tensor([0.0, 0.0, 1.0])
            E = self.edges_trunk

            def trunk_points(height):
                return _slice(v, E, torch.tensor([0.0, 0.0, height]), up)

            def scan(z_lo, z_hi, score, n=24, keep=None):
                best, best_z = None, z_lo
                for zz in np.linspace(z_lo, z_hi, n):
                    pts = trunk_points(float(zz))
                    if keep is not None:
                        pts = pts[keep(pts)]
                    if len(pts) < 3:
                        continue
                    s = score(pts)
                    if best is None or s > best:
                        best, best_z = s, float(zz)
                return best_z

            left = lambda p: p[:, 0] > 0
            midline = v[v[:, 0].abs() < 0.01]
            crotch = float(midline[:, 2].min())

            lm = {}
            # ISO natural waist: narrowest front-view width between pelvis and ribs.
            lm["waist"] = scan(z("spine04"), z("spine02"),
                               lambda p: -float(p[:, 0].max() - p[:, 0].min()))
            lm["waist_iliac"] = ILIAC_CREST_RATIO[body["sex"]] * H + ABOVE_CREST_M
            lm["waist_navel"] = float(v[self.navel_vertex, 2])
            # ISO hip: level of greatest buttock prominence (furthest back, +y).
            lm["hip"] = scan(crotch, z("spine04"), lambda p: float(p[:, 1].max()))
            # ISO chest/bust: level of greatest chest prominence (furthest forward, -y).
            lm["chest"] = scan(z("spine02") + 0.3 * (z("spine01") - z("spine02")),
                               z("spine01") + 0.5 * (z("clavicle.L") - z("spine01")),
                               lambda p: -float(p[:, 1].min()))
            # ISO thigh: horizontal, highest position on the thigh (just under the crotch).
            lm["thigh"] = crotch - 0.02
            # ISO calf: horizontal, greatest girth between knee and ankle.
            lm["calf"] = scan(z("foot.L") + 0.1 * (z("lowerleg01.L") - z("foot.L")),
                              z("foot.L") + 0.75 * (z("lowerleg01.L") - z("foot.L")),
                              lambda p: _hull_perimeter(p[:, :2]), keep=left)

            # Arm tapes go around the limb, so the plane is square to the bone.
            sh, el, wr = j["upperarm01.L"], j["lowerarm01.L"], j["wrist.L"]

            def arm_scan(a, b, t_lo, t_hi, sign, radius):
                axis = (b - a) / (b - a).norm()
                best, best_t = None, t_lo
                for t in np.linspace(t_lo, t_hi, 12):
                    g = self._limb_girth(v, a + float(t) * (b - a), axis, radius)
                    if g is None:
                        continue
                    s = sign * float(g)
                    if best is None or s > best:
                        best, best_t = s, float(t)
                return best_t

            # Relaxed upper arm: greatest girth along the middle of the upper arm.
            lm["bicep_t"] = arm_scan(sh, el, 0.35, 0.75, +1, 0.15)
            # Wrist: smallest girth at the end of the forearm.
            lm["wrist_t"] = arm_scan(el, wr, 0.80, 1.00, -1, 0.08)
            lm["crotch"] = crotch
        return lm

    def _limb_girth(self, v, point, axis, radius):
        pts = _slice(v, self.edges_left_arm, point, axis)
        pts = pts[(pts - point).norm(dim=1) < radius]
        if len(pts) < 3:
            return None
        return _hull_perimeter(_to_plane(pts, point, axis))

    def measure(self, body, landmarks=None):
        """Girths in millimetres (torch scalars, differentiable w.r.t. the vertices)."""
        lm = landmarks or self.landmarks(body)
        v, j = body["vertices"], body["joints"]
        up = torch.tensor([0.0, 0.0, 1.0])

        def horizontal(height, left_only=False):
            pts = _slice(v, self.edges_trunk, torch.tensor([0.0, 0.0, height]), up)
            if left_only:
                pts = pts[pts[:, 0] > 0]
            return _hull_perimeter(pts[:, :2])

        sh, el, wr = j["upperarm01.L"], j["lowerarm01.L"], j["wrist.L"]
        upper_axis = ((el - sh) / (el - sh).norm()).detach()
        fore_axis = ((wr - el) / (wr - el).norm()).detach()
        g = {
            "waist": horizontal(lm["waist"]),
            "waist_iliac": horizontal(lm["waist_iliac"]),
            "waist_navel": horizontal(lm["waist_navel"]),
            "hip": horizontal(lm["hip"]),
            "thigh": horizontal(lm["thigh"], left_only=True),
            "calf": horizontal(lm["calf"], left_only=True),
            "chest": horizontal(lm["chest"]),
            "bicep": self._limb_girth(v, (sh + lm["bicep_t"] * (el - sh)).detach(), upper_axis, 0.15),
            "wrist": self._limb_girth(v, (el + lm["wrist_t"] * (wr - el)).detach(), fore_axis, 0.08),
        }
        return {k: g[k] * 1000.0 for k in GIRTH_KEYS}


# ---------------------------------------------------------------------- geometry

def _unique_edges(faces):
    e = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    return torch.unique(torch.sort(e, dim=1).values, dim=0)


def _slice(v, edges, point, normal):
    """Points where the mesh edges cross a plane. Differentiable w.r.t. v."""
    d = (v - point) @ normal
    d0, d1 = d[edges[:, 0]], d[edges[:, 1]]
    hit = (d0 * d1) < 0
    d0, d1 = d0[hit], d1[hit]
    t = (d0 / (d0 - d1)).unsqueeze(1)
    a, b = v[edges[hit, 0]], v[edges[hit, 1]]
    return a + t * (b - a)


def _to_plane(pts, point, normal):
    """3D points on a plane -> 2D coordinates in that plane."""
    helper = torch.tensor([0.0, 0.0, 1.0]) if abs(float(normal[2])) < 0.9 else torch.tensor([1.0, 0.0, 0.0])
    u = torch.linalg.cross(normal, helper)
    u = u / u.norm()
    w = torch.linalg.cross(normal, u)
    rel = pts - point
    return torch.stack([rel @ u, rel @ w], dim=1)


def _hull_perimeter(p2):
    """Length of a tight tape around 2D points. scipy picks which points are on the
    hull; the length itself is computed in torch so it stays differentiable."""
    hull = ConvexHull(p2.detach().numpy())
    ring = p2[torch.as_tensor(hull.vertices, dtype=torch.long)]
    return (ring - ring.roll(-1, dims=0)).norm(dim=1).sum()


def _volume(v, faces):
    a, b, c = v[faces[:, 0]], v[faces[:, 1]], v[faces[:, 2]]
    return (torch.linalg.cross(a, b) * c).sum(dim=1).sum().abs() / 6.0


# ---------------------------------------------------------------------- file I/O

def measure_mesh_file(mesh_npz, height_cm, weight_kg, sex, age, out_json):
    """mesh.npz (stage 1 contract) -> measurements.json (stage 2 contract).

    Needs anny_params in the npz as {"dials": {...}, "local": {...}} because the tape
    positions are found from Anny's joints. Waist keeps all three definitions.
    """
    data = np.load(mesh_npz, allow_pickle=True)
    params = data["anny_params"].item()
    anny_body = AnnyBody()
    with torch.no_grad():
        body = anny_body.build(sex, age, height_cm, params["dials"], params.get("local"))
        girths = {k: round(float(x), 1) for k, x in anny_body.measure(body).items()}
    result = {
        "height_cm": float(height_cm), "weight_kg": float(weight_kg),
        "sex": sex, "age": int(age), "girths_mm": girths,
    }
    Path(out_json).write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    import sys
    h, w, s = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
    ab = AnnyBody()
    dials, body = ab.from_height_weight_sex(h, w, s)
    print("dials", {k: round(v, 3) for k, v in dials.items()})
    print("mass_kg", round(float(body["mass_kg"]), 1))
    lm = ab.landmarks(body)
    print("tape heights (cm)", {k: round(v * 100, 1) for k, v in lm.items() if not k.endswith("_t")})
    print("girths_mm", {k: round(float(x)) for k, x in ab.measure(body, lm).items()})
