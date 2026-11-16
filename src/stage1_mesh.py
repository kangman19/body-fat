"""Stage 1: photo(s) -> SAM 3D Body -> neutral-pose body -> Anny A-pose fit -> mesh.npz

Steps
  1. SAM 3D Body reads each photo and predicts the person's body SHAPE numbers
     (independent of how they are standing). With front + side photos the shape
     numbers are averaged.
  2. Meta's body model (MHR) is rebuilt from those shape numbers in its neutral standing
     pose and scaled to the user's height. This removes the pose in the photo.
  3. That neutral body is tape-measured (same landmark rules as stage 2).
  4. Anny is fitted to that neutral body in two passes (fit_dense):
       a. joints: Anny's arms and legs are bent until its shoulders, elbows, wrists,
          hips, knees and ankles sit on the MHR ones;
       b. surface + silhouette + weight: Anny's dials and its shape and muscle sliders
          are optimised against three things at once:
            - the SAM skin (distance both ways, plus matching surface direction). SAM
              cannot see depth in one photo and tends to invent a soft belly, so
              front-to-back distances count for little;
            - the person's silhouette in the photo (SilhouetteLoss): Anny is carried
              into the photo's pose and camera and must match the cut-out in overlap,
              in row-by-row width and along the bare-skin outline;
            - the user's weight, which is what pins down depth.
     Head, hands and feet are left out of the comparisons.
     Abdominal tautness is then set from the estimated body fat.
     From here on the rest of the pipeline only ever sees an Anny body.

Everything runs on the CPU. Meta's code sends tensors to "cuda" in a few places no
matter what device was requested; those calls are redirected here, without editing the
cloned repo. Photos are processed in memory and never written to disk. The rembg model
(176 MB) is downloaded once into models/rembg and runs locally after that.
"""
from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
from scipy.spatial import ConvexHull, cKDTree

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("U2NET_HOME", str(ROOT / "models" / "rembg"))
SAM_REPO = ROOT / "sam-3d-body"
CHECKPOINT = SAM_REPO / "checkpoints" / "sam-3d-body-dinov3"

import stage3_nhanes as stage3  # noqa: E402
from stage2_measure import (ABOVE_CREST_M, DEFINITION_LOCALS, DIAL_RANGES, ILIAC_CREST_RATIO,  # noqa: E402
                            MUSCLE_LOCALS, SHAPE_LOCALS, AnnyBody)
from stage4_morph import relief  # noqa: E402

FIT_KEYS = ["waist", "waist_iliac", "hip", "chest", "thigh", "calf", "bicep", "wrist"]


class SamBody:
    """SAM 3D Body on CPU. Loading takes about 40 s; each photo about 90 s."""

    def __init__(self):
        sys.path.insert(0, str(SAM_REPO))
        import sam_3d_body.models.meta_arch.sam3d_body as arch
        import sam_3d_body.sam_3d_body_estimator as est
        from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body
        from sam_3d_body.metadata.mhr70 import pose_info

        to_device = est.recursive_to
        est.recursive_to = arch.recursive_to = lambda x, target: to_device(x, "cpu")
        torch.Tensor.cuda = lambda self, *a, **k: self

        self.model, cfg = load_sam_3d_body(
            str(CHECKPOINT / "model.ckpt"), device="cpu",
            mhr_path=str(CHECKPOINT / "assets" / "mhr_model.pt"))
        self.estimator = SAM3DBodyEstimator(sam_3d_body_model=self.model, model_cfg=cfg)
        self.faces = np.asarray(self.estimator.faces)
        self.keypoint_names = {i["id"]: i["name"] for i in pose_info["keypoint_info"].values()}

    def release(self):
        """Free the model (about 3 GB). Only self.faces is usable afterwards."""
        self.model = self.estimator = None
        gc.collect()

    def shape_from_photos(self, images_rgb: list[np.ndarray]):
        """Average shape and scale numbers over the photos. The person is assumed to
        fill the photo (no person detector), so the whole image is used as the box.
        Also returns how the first (front) photo was seen, for painting it on later."""
        shapes, scales, view = [], [], None
        for img in images_rgb:
            h, w = img.shape[:2]
            people = self.estimator.process_one_image(
                img, bboxes=np.array([[0, 0, w, h]], dtype=np.float32), inference_type="body")
            if not people:
                raise ValueError("No person found in the photo.")
            shapes.append(np.asarray(people[0]["shape_params"], dtype=np.float32))
            scales.append(np.asarray(people[0]["scale_params"], dtype=np.float32))
            if view is None:
                view = {"vertices": np.asarray(people[0]["pred_vertices"], dtype=np.float64),
                        "cam_t": np.asarray(people[0]["pred_cam_t"], dtype=np.float64),
                        "focal": float(people[0]["focal_length"]), "size": (w, h)}
        return np.mean(shapes, axis=0), np.mean(scales, axis=0), view

    def neutral_body(self, shape, scale, height_cm):
        """MHR in its neutral pose, converted to Anny's axes (x left, y back, z up),
        feet on z = 0, scaled to height_cm. Returns (vertices, keypoints by name,
        scale factor applied to SAM's own metres)."""
        with torch.no_grad():
            v, kp = self.model.head_pose.mhr_forward(
                torch.zeros(1, 3), torch.zeros(1, 3), torch.zeros(1, 133), None,
                torch.as_tensor(scale)[None], torch.as_tensor(shape)[None],
                expr_params=torch.zeros(1, 72), return_keypoints=True)
        flip = lambda a: np.stack([a[:, 0], -a[:, 2], a[:, 1]], axis=1)  # MHR: y up, z forward
        v, kp = flip(v[0].numpy()), flip(kp[0].numpy()[:70])
        zmin = v[:, 2].min()
        s = (height_cm / 100.0) / (v[:, 2].max() - zmin)
        off = np.array([0, 0, zmin])
        return (v - off) * s, {self.keypoint_names[i]: (kp[i] - off) * s for i in range(70)}, float(s)


# ------------------------------------------------------ tape measure for any mesh

def _loops(mesh, origin, normal):
    """Separate closed outlines where a plane cuts the mesh (torso, each arm, each leg)."""
    seg = trimesh.intersections.mesh_plane(mesh, normal, origin)
    if len(seg) == 0:
        return []
    return [np.asarray(p) for p in trimesh.load_path(seg).discrete if len(p) >= 3]


def _girth(points, normal):
    helper = np.array([0, 0, 1.0]) if abs(normal[2]) < 0.9 else np.array([1.0, 0, 0])
    u = np.cross(normal, helper); u /= np.linalg.norm(u)
    w = np.cross(normal, u)
    p2 = np.stack([points @ u, points @ w], axis=1)
    hull = p2[ConvexHull(p2).vertices]
    return float(np.linalg.norm(hull - np.roll(hull, -1, axis=0), axis=1).sum())


def measure_generic(vertices, faces, kp, sex) -> dict:
    """Girths in mm on a neutral-pose mesh, using keypoints to find the tape positions.
    Same definitions as stage2_measure (no navel waist: this mesh has no navel marker)."""
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    H = float(vertices[:, 2].max())
    up = np.array([0, 0, 1.0])
    z = lambda name: float(kp[name][2])
    shoulder_x = float(kp["left_shoulder"][0])

    def torso(height):  # outlines centred on the body, i.e. not an arm or hand
        loops = [l for l in _loops(mesh, np.array([0, 0, height]), up)
                 if abs(l[:, 0].mean()) < 0.6 * shoulder_x]
        return np.concatenate(loops) if loops else None

    def left_leg(height):
        loops = [l for l in _loops(mesh, np.array([0, 0, height]), up)
                 if 0 < l[:, 0].mean() < 2.5 * shoulder_x and l[:, 0].min() > -0.02]
        return min(loops, key=lambda l: abs(l[:, 0].mean() - kp["left_hip"][0])) if loops else None

    def scan(lo, hi, getter, score, n=24):
        best = None
        for h in np.linspace(lo, hi, n):
            pts = getter(float(h))
            if pts is None:
                continue
            s = score(pts)
            if best is None or s > best[0]:
                best = (s, pts)
        if best is None:
            raise ValueError("Could not measure the body from this photo.")
        return best[1]

    # Crotch: going down from the hip joints, the first height with no central outline.
    crotch = z("left_hip") - 0.15 * H
    for h in np.arange(z("left_hip"), z("left_knee"), -0.005):
        if torso(float(h)) is None:
            crotch = float(h)
            break

    hip_z, sh_z = z("left_hip"), z("left_shoulder")
    chest_lo = hip_z + 0.55 * (sh_z - hip_z)
    g = {}
    g["waist"] = _girth(scan(hip_z + 0.05 * H, chest_lo, torso,
                             lambda p: -(p[:, 0].max() - p[:, 0].min())), up)
    iliac = torso(ILIAC_CREST_RATIO[sex] * H + ABOVE_CREST_M)
    g["waist_iliac"] = _girth(iliac, up) if iliac is not None else g["waist"]
    g["hip"] = _girth(scan(crotch + 0.005, hip_z + 0.06 * H, torso, lambda p: p[:, 1].max()), up)
    g["chest"] = _girth(scan(chest_lo, hip_z + 0.82 * (sh_z - hip_z), torso,
                             lambda p: -p[:, 1].min()), up)
    thigh = left_leg(crotch - 0.02)
    if thigh is None:
        raise ValueError("Could not measure the thigh from this photo.")
    g["thigh"] = _girth(thigh, up)
    ankle, knee = z("left_ankle"), z("left_knee")
    g["calf"] = _girth(scan(ankle + 0.1 * (knee - ankle), ankle + 0.75 * (knee - ankle),
                            left_leg, lambda p: _girth(p, up)), up)

    def arm(a, b, t_lo, t_hi, sign):
        axis = (b - a) / np.linalg.norm(b - a)
        best = None
        for t in np.linspace(t_lo, t_hi, 12):
            origin = a + t * (b - a)
            loops = _loops(mesh, origin, axis)
            if not loops:
                continue
            loop = min(loops, key=lambda l: np.linalg.norm(l.mean(axis=0) - origin))
            girth = _girth(loop, axis)
            if best is None or sign * girth > sign * best:
                best = girth
        if best is None:
            raise ValueError("Could not measure the arm from this photo.")
        return best

    g["bicep"] = arm(kp["left_shoulder"], kp["left_elbow"], 0.35, 0.75, +1)
    g["wrist"] = arm(kp["left_elbow"], kp["left_wrist"], 0.80, 1.00, -1)
    return {k: v * 1000.0 for k, v in g.items()}


# ------------------------------------------------------------------- full stage

def decode_image(data: bytes) -> np.ndarray:
    """Image file bytes -> RGB array (phone rotation tags are applied by OpenCV)."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("That file is not an image this app can read.")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def person_mask(image_rgb) -> np.ndarray:
    """True where the person is. The model is loaded for this call and then released."""
    from rembg import new_session, remove
    return np.asarray(remove(image_rgb, session=new_session("u2net_human_seg"), only_mask=True)) > 127


def _body_keep(vertices, kp):
    """MHR vertices that are not head, hands or feet (neutral pose, Anny axes)."""
    keep = (vertices[:, 2] < kp["neck"][2] + 0.03) & \
           (vertices[:, 2] > max(kp["left_ankle"][2], kp["right_ankle"][2]))
    for side in ("left", "right"):
        wrist, elbow = kp[side + "_wrist"], kp[side + "_elbow"]
        axis = (wrist - elbow) / np.linalg.norm(wrist - elbow)
        keep &= ~(((vertices - wrist) @ axis > -0.01) & (np.linalg.norm(vertices - wrist, axis=1) < 0.25))
    return keep


def _vertex_frames(vertices, faces):
    """A small coordinate frame glued to the skin at every vertex: (V, 3, 3)."""
    n = np.asarray(trimesh.Trimesh(vertices, faces, process=False).vertex_normals)
    other = np.zeros(len(vertices), dtype=np.int64)
    for i in range(3):
        other[faces[:, i]] = faces[:, (i + 1) % 3]
    e = vertices[other] - vertices
    t = e - (e * n).sum(1, keepdims=True) * n
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-12
    return np.stack([t, np.cross(n, t), n], axis=2)


class PhotoPose:
    """Carries a shape defined in the neutral pose into the pose and camera of the photo.

    SAM gives the same MHR mesh twice: neutral (N) and posed over the photo (P). For any
    other skin near the neutral one, each MHR vertex takes the offset to that skin,
    turns it by the rotation its own patch of skin went through from N to P, and adds
    it to P. That gives the other body as it would appear in the photo.
    """

    def __init__(self, view, neutral_vertices, faces, scale, keep):
        self.view, self.scale, self.faces = view, scale, faces
        self.N = torch.tensor(neutral_vertices, dtype=torch.float32)
        self.P = torch.tensor(view["vertices"] + view["cam_t"], dtype=torch.float32)
        frames_n, frames_p = _vertex_frames(neutral_vertices, faces), _vertex_frames(view["vertices"], faces)
        self.R = torch.tensor(frames_p @ frames_n.transpose(0, 2, 1), dtype=torch.float32)
        self.keep = torch.as_tensor(keep)
        self.idx = None

    def correspond(self, skin_np):
        """Pair every MHR vertex with its nearest point of the other skin (neutral pose)."""
        self.idx = torch.as_tensor(cKDTree(skin_np).query(self.N.numpy())[1])

    def carry(self, skin):
        """skin: (V_other, 3) torch, neutral pose. Returns camera-space MHR-layout points."""
        offset = (skin[self.idx] - self.N) / self.scale
        return self.P + (self.R @ offset[:, :, None])[:, :, 0]

    def project(self, points):
        w, h = self.view["size"]
        f = self.view["focal"]
        return torch.stack([f * points[:, 0] / points[:, 2] + w / 2,
                            f * points[:, 1] / points[:, 2] + h / 2], dim=1)


def skin_mask(image_rgb, mask, photo_pose, kp):
    """Bare skin inside the person cut-out, found by colour relative to the subject's
    own face, so it works for any skin tone. Returns (skin bool HxW, face colour RGB),
    or (None, None) when the face cannot be sampled (hidden or cropped out).

    Everything in the cut-out that is not skin is treated as clothing, hair or
    accessories. (A clothing segmenter was tried and rejected: it labelled a bare torso
    as an upper garment.)
    """
    N = photo_pose.N.numpy()
    normals = np.asarray(trimesh.Trimesh(N, photo_pose.faces, process=False).vertex_normals)
    nose = kp["nose"]
    face = (np.abs(N[:, 2] - nose[2]) < 0.055) & (np.abs(N[:, 0]) < 0.055) & (normals[:, 1] < -0.5)
    px = photo_pose.project(photo_pose.P[torch.as_tensor(face)]).numpy()
    h, w = mask.shape
    core = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=3) > 0
    xi, yi = np.round(px[:, 0]).astype(int), np.round(px[:, 1]).astype(int)
    ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    ok[ok] &= core[yi[ok], xi[ok]]
    if ok.sum() < 50:
        return None, None
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    sample = lab[yi[ok], xi[ok]]
    # Skin of any tone is warm: its colour sits away from neutral grey in a consistent
    # direction (the a/b axes of Lab), in light and in shade. White, grey and black
    # cloth sit on neutral, so "close to the face colour" is not enough for dark or very
    # pale skin. Require the same direction of tint and at least 45% of its strength.
    tint = np.median(sample[:, 1:], axis=0) - 128.0
    strength = float(np.linalg.norm(tint))
    if strength < 4.0:      # a face with no measurable tint (monochrome photo): cannot tell
        return None, None
    unit = tint / strength
    chroma = lab[..., 1:] - 128.0
    along = chroma @ unit
    across = np.abs(chroma @ np.array([-unit[1], unit[0]]))
    skin = mask & (along > 0.45 * strength) & (across < 0.6 * strength + 4.0)
    skin = cv2.morphologyEx(skin.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)) > 0
    return skin & mask, np.median(image_rgb[yi[ok], xi[ok]], axis=0)


class SilhouetteLoss:
    """How well a body, carried into the photo's pose and camera, matches the person's
    cut-out. Four differentiable terms:

      inside  no body point may fall outside the cut-out (always valid: skin or cloth,
              the body is inside it);
      edge    every trusted outline pixel, including the gaps between arm and torso,
              should have the body's surface on it;
      width   row by row down the photo, the outermost left and right body points
              should sit on the outermost cut-out pixels (shoulders, arms, waist, legs);
      iou     soft silhouette overlap: the body is splatted to a coarse image and
              compared with the cut-out as intersection over union.

    Only bare skin is trusted to pull the body outwards. Where the cut-out's edge is
    clothing or hair, or is the photo's own border (a cropped body), only `inside`
    applies and the SAM body and the weight decide. If skin cannot be detected, the
    outline is trusted only where SAM's own body came within 2 cm of it.
    Head, hands and feet are never fitted.
    """

    GRID = 256

    def __init__(self, photo_pose: "PhotoPose", mask, skin, anny_body: AnnyBody):
        self.pp = pp = photo_pose
        h, w = mask.shape
        self.size = (w, h)
        self.faces = anny_body.faces
        self.body = anny_body.fit_mask                      # torch bool per vertex
        N = pp.N.numpy()
        self.tree = cKDTree(np.hstack([N, 0.05 * np.asarray(
            trimesh.Trimesh(N, pp.faces, process=False).vertex_normals)]))
        self.match = None

        self.outside = torch.tensor(cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5),
                                    dtype=torch.float32)[None, None]
        if skin is not None:
            # skin right at the edge is mixed with background, so look a few pixels inwards
            trusted = cv2.dilate(skin.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=6) > 0
        else:  # fall back to "where SAM agreed with the outline"
            sam_px = np.round(pp.project(pp.P).numpy()).astype(np.int32)
            sam = np.zeros((h, w), np.uint8)
            for tri in pp.faces:
                cv2.fillConvexPoly(sam, sam_px[tri], 1)
            mpp = float(pp.P[:, 2].median()) / pp.view["focal"]
            trusted = cv2.distanceTransform(1 - sam, cv2.DIST_L2, 5) * mpp < 0.02
        margin = 3
        trusted[:margin] = trusted[-margin:] = False        # the photo's border is not a body edge
        trusted[:, :margin] = trusted[:, -margin:] = False
        self.trusted = trusted

        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        pts = np.concatenate([c[:, 0, :] for c in contours])
        self.outline = torch.tensor(pts[trusted[pts[:, 1], pts[:, 0]]], dtype=torch.float32)
        self.trusted_share = round(float(len(self.outline) / max(1, len(pts))), 2)

        # rows for the width term
        self.step = max(3, h // 120)
        rows = np.arange(self.step, h - self.step, self.step)
        rows = np.array([y for y in rows if mask[y].any()])
        left = np.array([np.nonzero(mask[y])[0].min() for y in rows])
        right = np.array([np.nonzero(mask[y])[0].max() for y in rows])
        self.rows = torch.tensor(rows, dtype=torch.float32)
        self.left, self.right = torch.tensor(left, dtype=torch.float32), torch.tensor(right, dtype=torch.float32)
        self.left_ok = torch.as_tensor(trusted[rows, left])
        self.right_ok = torch.as_tensor(trusted[rows, right])
        self.row_side_ok = None

        # coarse image for the overlap term
        g = self.GRID / max(w, h)
        self.g, self.gw, self.gh = g, int(round(w * g)), int(round(h * g))
        small = lambda a: torch.tensor(cv2.resize(a.astype(np.float32), (self.gw, self.gh),
                                                  interpolation=cv2.INTER_AREA))
        self.M = small(mask)
        self.W = small(~mask | trusted)          # untrusted (clothed) interior asks for nothing
        k = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0
        self.blur = (k[:, None] * k[None, :])[None, None]
        self.mask_np = mask

    def correspond(self, vertices_np, normals_np):
        """Pair every Anny vertex with a point of the neutral SAM body (position and
        surface direction), so it can be carried into the photo."""
        self.match = torch.as_tensor(self.tree.query(np.hstack([vertices_np, 0.05 * normals_np]))[1])

    def carry(self, vertices):
        """(V, 3) Anny vertices in the fit pose -> camera-space points in the photo."""
        pp, i = self.pp, self.match
        return pp.P[i] + (pp.R[i] @ ((vertices - pp.N[i]) / pp.scale)[:, :, None])[:, :, 0]

    def _dense(self, px):
        """Vertices plus edge midpoints and triangle centres, so sparse areas have no holes."""
        f = self.faces
        a, b, c = px[f[:, 0]], px[f[:, 1]], px[f[:, 2]]
        return torch.cat([px, (a + b) / 2, (b + c) / 2, (c + a) / 2, (a + b + c) / 3])

    def _dense_flag(self, flag):
        f = self.faces
        both = lambda i, j: flag[f[:, i]] & flag[f[:, j]]
        return torch.cat([flag, both(0, 1), both(1, 2), both(2, 0), both(0, 1) & flag[f[:, 2]]])

    def __call__(self, vertices, refresh: bool):
        cam = self.carry(vertices)
        px = self.pp.project(cam)
        f = self.pp.view["focal"]
        mpp = (cam[:, 2] / f).detach()
        scale = float(mpp.median())
        w, h = self.size
        body = self.body

        grid = torch.stack([px[:, 0] / (w - 1) * 2 - 1, px[:, 1] / (h - 1) * 2 - 1], dim=1)[None, None]
        out_px = torch.nn.functional.grid_sample(self.outside, grid, align_corners=True,
                                                 padding_mode="border")[0, 0, 0]
        inside = ((out_px * mpp)[body] ** 2).mean()

        dense = self._dense(px)
        dense_body = self._dense_flag(body)
        if refresh or self.row_side_ok is None:
            d_np = dense.detach().numpy()
            dist, j = cKDTree(d_np).query(self.outline.numpy())
            j = torch.as_tensor(j)
            self.edge_pair = (j, dense_body[j] & (torch.as_tensor(dist) * scale < 0.05))
            # which rows have a fitted body part (not head, hand or foot) at their extremes
            r = np.round((d_np[:, 1] - float(self.rows[0])) / self.step).astype(int)
            n = len(self.rows)
            left_ok, right_ok = np.zeros(n, bool), np.zeros(n, bool)
            valid = (r >= 0) & (r < n)
            order = np.lexsort((d_np[valid, 0], r[valid]))
            rs, flags = r[valid][order], dense_body.numpy()[valid][order]
            first = np.r_[True, rs[1:] != rs[:-1]]; last = np.r_[rs[1:] != rs[:-1], True]
            left_ok[rs[first]], right_ok[rs[last]] = flags[first], flags[last]
            self.row_side_ok = (torch.as_tensor(left_ok) & self.left_ok, torch.as_tensor(right_ok) & self.right_ok)
        j, ok = self.edge_pair
        gap = (dense[j] - self.outline)[ok] * scale
        edge = (gap ** 2).sum(1).mean() if ok.any() else torch.zeros(())

        # width: soft leftmost / rightmost body point in every row
        n, tau = len(self.rows), 0.5
        r = torch.round((dense[:, 1].detach() - self.rows[0]) / self.step).long()
        valid = (r >= 0) & (r < n)
        r, x = r[valid], dense[valid, 0]

        def soft_max(values):
            top = torch.full((n,), -1e9).scatter_reduce(0, r, values.detach(), "amax", include_self=True)
            total = torch.zeros(n).index_add(0, r, torch.exp(values - top[r]))
            return top + torch.log(total.clamp_min(1e-30))

        right, left = tau * soft_max(x / tau), -tau * soft_max(-x / tau)
        left_ok, right_ok = self.row_side_ok
        res = torch.cat([(left - self.left)[left_ok], (right - self.right)[right_ok]]) * scale
        res = res[res.abs() < 0.06]             # further than 6 cm is a pose mismatch, not a width
        width = (res ** 2).mean() if len(res) else torch.zeros(())

        # soft overlap
        src = torch.where(dense_body[:, None], dense, dense.detach()) * self.g
        x0, y0 = torch.floor(src[:, 0]).detach(), torch.floor(src[:, 1]).detach()
        fx, fy = src[:, 0] - x0, src[:, 1] - y0
        density = torch.zeros(self.gh * self.gw)
        for dx, dy, wt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            xi, yi = (x0 + dx).long(), (y0 + dy).long()
            keep = (xi >= 0) & (xi < self.gw) & (yi >= 0) & (yi < self.gh)
            density = density.index_add(0, (yi * self.gw + xi)[keep], wt[keep])
        density = torch.nn.functional.conv2d(density.view(1, 1, self.gh, self.gw), self.blur, padding=2)[0, 0]
        S = 1 - torch.exp(-density)
        iou = (self.W * S * self.M).sum() / (self.W * (S + self.M - S * self.M)).sum()
        return {"inside": inside, "edge": edge, "width": width, "iou": 1 - iou}

    def report(self, vertices) -> dict:
        """Plain numbers for the final body: true (hard) overlap with the whole cut-out,
        and how far its drawn edge is from the trusted skin outline."""
        with torch.no_grad():
            px = np.round(self.pp.project(self.carry(vertices)).numpy()).astype(np.int32)
            scale_mm = float((self.carry(vertices)[:, 2] / self.pp.view["focal"]).median()) * 1000
        w, h = self.size
        drawn = np.zeros((h, w), np.uint8)
        for tri in self.faces.numpy():
            cv2.fillConvexPoly(drawn, px[tri], 1)
        edge = cv2.morphologyEx(drawn, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        to_edge = cv2.distanceTransform(1 - edge, cv2.DIST_L2, 5)
        o = self.outline.numpy().astype(int)
        gaps = to_edge[o[:, 1], o[:, 0]] * scale_mm
        gaps = gaps[gaps < 60]
        drawn = drawn > 0
        rows = self.rows.numpy().astype(int)
        widths = []
        for y, l, r, lo, ro in zip(rows, self.left.numpy(), self.right.numpy(), self.left_ok.numpy(), self.right_ok.numpy()):
            xs = np.nonzero(drawn[y])[0]
            if lo and ro and len(xs):
                widths.append(((xs.max() - xs.min()) - (r - l)) * scale_mm)
        return {"iou": round(float((drawn & self.mask_np).sum() / (drawn | self.mask_np).sum()), 3),
                "skin_edge_gap_mm": round(float(np.mean(gaps)), 1) if len(gaps) else None,
                "width_error_mm": round(float(np.mean(widths)), 1) if widths else None,
                "width_abs_error_mm": round(float(np.mean(np.abs(widths))), 1) if widths else None,
                "trusted_outline_share": self.trusted_share}


POSE_BONES = [b + side for b in ("upperarm01", "lowerarm01", "upperleg01", "lowerleg01")
              for side in (".L", ".R")]
JOINT_TO_KEYPOINT = {"upperarm01": "shoulder", "lowerarm01": "elbow", "wrist": "wrist",
                     "upperleg01": "hip", "lowerleg01": "knee", "foot": "ankle"}
FITTED_LOCALS = SHAPE_LOCALS + MUSCLE_LOCALS


def _vertex_normals(v, faces):
    a, b, c = v[faces[:, 0]], v[faces[:, 1]], v[faces[:, 2]]
    fn = torch.linalg.cross(b - a, c - a)
    vn = torch.zeros_like(v)
    for i in range(3):
        vn = vn.index_add(0, faces[:, i], fn)
    return vn / (vn.norm(dim=1, keepdim=True) + 1e-12)


def fit_dense(anny_body: AnnyBody, target_v, target_faces, kp, sex, age, height_cm, weight_kg,
              silhouette: SilhouetteLoss | None = None, joint_steps=120, surface_steps=450, log=None):
    """Fit Anny to a neutral-pose target mesh and, if given, the photo silhouette.
    Returns a dict with dials, local, pose, trans, surface_error_mm and report."""
    target = trimesh.Trimesh(target_v, target_faces, process=False)
    keep = _body_keep(target_v, kp)
    T = torch.tensor(target_v[keep], dtype=torch.float32)
    Tn = torch.tensor(np.asarray(target.vertex_normals)[keep], dtype=torch.float32)
    # One photo shows width, not depth: trust the SAM skin less where it faces front/back.
    trust = 1.0 - 0.85 * Tn[:, 1].abs()
    tree_T = cKDTree(T.numpy())
    mask, faces = anny_body.fit_mask, anny_body.faces
    joint_targets = {bone + s: torch.tensor(kp[side + "_" + name], dtype=torch.float32)
                     for bone, name in JOINT_TO_KEYPOINT.items()
                     for s, side in ((".L", "left"), (".R", "right"))}

    start, _ = anny_body.from_height_weight_sex(height_cm, weight_kg, sex, age)
    logit = lambda k: float(np.log((x := min(max((start[k] - DIAL_RANGES[k][0]) /
                            (DIAL_RANGES[k][1] - DIAL_RANGES[k][0]), 1e-4), 1 - 1e-4)) / (1 - x)))
    raw_d = {k: torch.tensor(logit(k), requires_grad=True) for k in DIAL_RANGES}
    raw_l = {k: torch.zeros((), requires_grad=True) for k in FITTED_LOCALS}
    # Muscle sliders may go past Anny's nominal range: athletes exist. They are also the
    # cheapest to move, so width is explained by muscle before it is explained by
    # stretching. Depth sliders cost most: one front photo cannot see depth, so depth
    # should follow from build and weight.
    reach = {k: (1.5 if k in MUSCLE_LOCALS else 1.0) for k in FITTED_LOCALS}
    cost = {k: (0.25 if k in MUSCLE_LOCALS else 5.0 if "depth" in k else 1.0) for k in FITTED_LOCALS}
    pose = {b: torch.zeros(3, requires_grad=True) for b in POSE_BONES}
    trans = torch.zeros(3, requires_grad=True)

    def current():
        dials = {k: DIAL_RANGES[k][0] + (DIAL_RANGES[k][1] - DIAL_RANGES[k][0]) * torch.sigmoid(r)
                 for k, r in raw_d.items()}
        return dials, {k: reach[k] * torch.tanh(r) for k, r in raw_l.items()}

    def joints_loss(body):
        return torch.stack([((body["posed_joints"][b] + trans - t) ** 2).sum()
                            for b, t in joint_targets.items()]).mean()

    with torch.no_grad():  # start with the two skeletons roughly on top of each other
        dials, local = current()
        body = anny_body.build(sex, age, height_cm, dials, local, pose)
        trans += torch.stack([t - body["posed_joints"][b] for b, t in joint_targets.items()]).mean(0)

    # Pass a: joints only, bending limbs and sliding the body.
    opt = torch.optim.Adam(list(pose.values()) + [trans], lr=0.02)
    for step in range(joint_steps):
        opt.zero_grad()
        dials, local = current()
        loss = joints_loss(anny_body.build(sex, age, height_cm, dials, local, pose))
        loss.backward(); opt.step()
        if log and step % 40 == 0:
            log(f"joints  {step:3d}  mean joint gap {float(loss.detach()) ** 0.5 * 1000:6.1f} mm")

    # Pass b: surface, silhouette and weight together.
    opt = torch.optim.Adam([
        {"params": list(raw_d.values()) + list(raw_l.values()), "lr": 0.04},
        {"params": list(pose.values()) + [trans], "lr": 0.008}])
    gap, photo = None, {}
    for step in range(surface_steps):
        opt.zero_grad()
        dials, local = current()
        body = anny_body.build(sex, age, height_cm, dials, local, pose)
        pv = body["posed_vertices"] + trans
        normals = _vertex_normals(pv, faces)
        A, An = pv[mask], normals[mask]
        refresh = step % 5 == 0
        if refresh:  # refresh which point pairs with which
            a_np = A.detach().numpy()
            dist_T, idx_T = tree_T.query(a_np)
            dist_A, idx_A = cKDTree(a_np).query(T.numpy())
            idx_T, idx_A = torch.as_tensor(idx_T), torch.as_tensor(idx_A)
            ok_T = torch.as_tensor(dist_T < 0.06) & ((An.detach() * Tn[idx_T]).sum(1) > 0)
            ok_A = torch.as_tensor(dist_A < 0.06) & ((An.detach()[idx_A] * Tn).sum(1) > 0)
            if silhouette is not None:
                silhouette.correspond(pv.detach().numpy(), normals.detach().numpy())
        d1, d2 = (A - T[idx_T])[ok_T], (A[idx_A] - T)[ok_A]
        plane1, plane2 = (d1 * Tn[idx_T][ok_T]).sum(1), (d2 * Tn[ok_A]).sum(1)
        t1, t2 = trust[idx_T][ok_T], trust[ok_A]
        surface = (t1 * plane1 ** 2).mean() + (t2 * plane2 ** 2).mean() + \
            0.1 * ((t1 * (d1 ** 2).sum(1)).mean() + (t2 * (d2 ** 2).sum(1)).mean())
        direction = (1 - (An * Tn[idx_T]).sum(1))[ok_T].mean()
        loss = 3e3 * surface + 0.5 * direction + 2e2 * joints_loss(body) \
            + torch.stack([cost[k] * (v / reach[k]) ** 2 for k, v in local.items()]).mean() \
            + 200.0 * ((body["mass_kg"] - weight_kg) / weight_kg) ** 2 \
            + 10.0 * (body["natural_height_m"] - height_cm / 100.0) ** 2
        if silhouette is not None:
            photo = silhouette(pv, refresh)
            loss = loss + 1e4 * (photo["inside"] + photo["edge"] + photo["width"]) + 25.0 * photo["iou"]
        loss.backward(); opt.step()
        gap = float((plane1.abs().mean() + plane2.abs().mean()).detach() / 2) * 1000
        if log and (step % 50 == 0 or step == surface_steps - 1):
            extra = "".join(f"  {k} {float(v.detach()) ** (1 if k == 'iou' else 0.5) * (1 if k == 'iou' else 1000):5.2f}"
                            for k, v in photo.items())
            log(f"surface {step:3d}  skin gap {gap:5.2f} mm{extra}  weight {float(body['mass_kg'].detach()):5.1f} kg")

    with torch.no_grad():
        dials, local = current()
        dials = {k: float(v) for k, v in dials.items()}
        local = {k: float(v) for k, v in local.items()}
        report = {}
        if silhouette is not None:
            final = anny_body.build(sex, age, height_cm, dials, local, pose)
            report = silhouette.report(final["posed_vertices"] + trans)
    return {"dials": dials, "local": local, "pose": {b: r.detach() for b, r in pose.items()},
            "trans": trans.detach(), "surface_error_mm": gap, "report": report}


def fit_from_photos(sam: SamBody, anny_body: AnnyBody, bf_model, images_rgb, height_cm,
                    weight_kg, sex, age, log=None) -> dict:
    """Photos -> fitted Anny body. `sam` is released part-way through, so pass a fresh
    SamBody() each time. Returns a dict:
    dials, local (shape, muscle and definition sliders), body (rest pose),
    photo_girths_mm (tape on the SAM body), bf_mid, surface_error_mm, report (silhouette
    match), plus what stage 5 needs to draw the photo on the body: mask, skin,
    photo_pose, posed_vertices (fit pose), pose, trans."""
    shape, scale, view = sam.shape_from_photos(images_rgb)
    v, kp, factor = sam.neutral_body(shape, scale, height_cm)
    sam.release()  # the big model is not needed again; make room for the next steps
    photo_girths = measure_generic(v, sam.faces, kp, sex)
    mask = person_mask(images_rgb[0])
    photo_pose = PhotoPose(view, v, sam.faces, factor, _body_keep(v, kp))
    skin, face_colour = skin_mask(images_rgb[0], mask, photo_pose, kp)
    fit = fit_dense(anny_body, v, sam.faces, kp, sex, age, height_cm, weight_kg,
                    silhouette=SilhouetteLoss(photo_pose, mask, skin, anny_body), log=log)
    with torch.no_grad():
        smooth = anny_body.build(sex, age, height_cm, fit["dials"], fit["local"])
        girths = {k: float(x) for k, x in anny_body.measure(smooth).items()}
        bf = stage3.estimate_bf(bf_model, sex, age, height_cm, float(smooth["mass_kg"]), girths)["bf_mid"]
        local = {**fit["local"], **{k: relief(sex, bf) for k in DEFINITION_LOCALS}}
        # The definition sliders change volume slightly, so re-set the fat dial for the weight.
        lo, hi = DIAL_RANGES["weight"]
        for _ in range(22):
            fit["dials"]["weight"] = (lo + hi) / 2
            if float(anny_body.build(sex, age, height_cm, fit["dials"], local)["mass_kg"]) < weight_kg:
                lo = fit["dials"]["weight"]
            else:
                hi = fit["dials"]["weight"]
        body = anny_body.build(sex, age, height_cm, fit["dials"], local, fit["pose"])
        posed = (body["posed_vertices"] + fit["trans"]).numpy()
    return {"dials": fit["dials"], "local": local, "body": body, "photo_girths_mm": photo_girths,
            "bf_mid": bf, "surface_error_mm": fit["surface_error_mm"], "report": fit["report"],
            "view": view, "neutral": (v, kp), "pose": fit["pose"], "trans": fit["trans"],
            "mask": mask, "skin": skin, "face_colour": face_colour, "photo_pose": photo_pose,
            "posed_vertices": posed}


def save_mesh_npz(path, anny_body: AnnyBody, body, dials, local):
    """mesh.npz contract (stage 1 output)."""
    np.savez(path, vertices=body["vertices"].detach().numpy().astype(np.float32),
             faces=anny_body.faces.numpy().astype(np.int32),
             anny_params=np.array({"dials": dials, "local": local}, dtype=object))


if __name__ == "__main__":
    # py src/stage1_mesh.py <height_cm> <weight_kg> <sex> <age> <photo> [<photo> ...]
    h, w, sex, age = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3], float(sys.argv[4])
    images = [decode_image(Path(p).read_bytes()) for p in sys.argv[5:]]
    ab = AnnyBody()
    r = fit_from_photos(SamBody(), ab, stage3.load_model(), images, h, w, sex, age, log=print)
    with torch.no_grad():
        fitted = {k: float(x) for k, x in ab.measure(r["body"]).items()}
    print(f"\nmean skin gap {r['surface_error_mm']:.1f} mm   estimated body fat {r['bf_mid']:.1f}%")
    print(f"{'':12s}{'SAM body':>12s}{'Anny fit':>12s}{'gap':>8s}")
    for k in FIT_KEYS:
        print(f"{k:12s}{r['photo_girths_mm'][k]:12.0f}{fitted[k]:12.0f}{fitted[k] - r['photo_girths_mm'][k]:8.0f}")
    print(f"{'weight kg':12s}{w:12.1f}{float(r['body']['mass_kg']):12.1f}")
    out = ROOT / "outputs" / "stage1"
    out.mkdir(parents=True, exist_ok=True)
    save_mesh_npz(out / "mesh.npz", ab, r["body"], r["dials"], r["local"])
    print("saved", out / "mesh.npz")
