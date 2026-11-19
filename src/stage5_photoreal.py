"""Stage 5: appearance. Muscle relief on a clay body, and the photoreal 2D view.

Nothing here is tuned to one photo: every position is taken from Anny's own joints and
landmarks, and every colour from the uploaded image.

1. Relief (all bodies)
   Muscle contours are geometry and shading, never paint. One height field, defined on
   Anny's surface from its joints, describes the rectus abdominis blocks, linea alba,
   side lines, obliques, serratus, the lower pectoral border and sternum, the deltoid,
   and the quadriceps. It is delivered two ways from the same data so they agree:
     - a displacement map: the viewer subdivides the mesh twice and moves the vertices;
     - a tangent-space normal map, with slopes in real units, for the lighting.
   Its strength is stage4_morph.relief(sex, body fat): 0 above 25% (men) / 32% (women),
   full at 8% / 18%. The colour map is never darkened to fake a groove.

2. The 3D viewer draws every body, photo builds included, as plain matte clay: no
   photo is painted onto the mesh. The photo only appears in the 2D view below.

3. Photoreal 2D view (photo builds)
   The photo is split into layers by the fitted mesh: a background with the person
   removed (LaMa), the body (trunk, legs, head), each arm, and the locked head. The body
   follows the morphed mesh pixel by pixel; each arm moves whole, so a wider waist never
   squashes it and a slimmer one never tears it. Changes of shape and of muscle relief
   are lit with lighting estimated from the photo itself (brightness against surface
   direction on bare skin). Depth, normal, silhouette, albedo and head lock passes come
   from the same drawing (PhotoView.render(passes=True)).

Static data (UV raster, subdivision) is cached in models/anny_uv_<size>_v2.npz.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree

from stage2_measure import AnnyBody, default_dials

ROOT = Path(__file__).resolve().parents[1]
SIZE = 1024
ERODE_PX = 4                  # cut-out is shrunk by this much before skin light is measured
RELIEF_RANGE_M = 0.005        # height map stores -5 mm .. +5 mm
NORMAL_SCALE_FULL = 0.45      # viewer normalScale at full relief (slopes are baked to match)
PARTS = ["torso", "neck", "head", "upperarm", "lowerarm", "hand", "upperleg", "lowerleg", "foot"]


# ------------------------------------------------------------------ UV layout

def _midpoint_subdivide(faces: np.ndarray, n: int):
    """Split every triangle into four. Returns (faces, parents): each vertex of the new
    mesh is the average of its two parents (an old vertex is its own parent twice)."""
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    unique, inverse = np.unique(edges, axis=0, return_inverse=True)
    ab, bc, ca = (n + inverse.reshape(3, -1)).astype(np.int64)
    a, b, c = faces.T
    new = np.concatenate([np.stack([a, ab, ca], 1), np.stack([b, bc, ab], 1),
                          np.stack([c, ca, bc], 1), np.stack([ab, bc, ca], 1)])
    parents = np.concatenate([np.repeat(np.arange(n)[:, None], 2, axis=1), unique])
    return new, parents


class UVLayout:
    """Anny's texture layout, plus the finer mesh the viewer draws.

    Texture seams mean one mesh vertex can have several UV points, so drawing starts
    from an 'expanded' mesh with one vertex per UV point (uv_to_vertex maps back). That
    mesh is then split twice (16 triangles per original) so displacement has enough
    vertices to form small muscle contours.
    """

    def __init__(self, anny_body: AnnyBody, size: int = SIZE):
        m = anny_body.model
        self.size = size
        self.uv = m.texture_coordinates.detach().numpy().astype(np.float32)        # (U, 2)
        self.uv_faces = m.face_texture_coordinate_indices.detach().numpy().astype(np.int64)  # (F, 3)
        self.faces = anny_body.faces.numpy()
        self.uv_to_vertex = np.zeros(len(self.uv), dtype=np.int64)
        self.uv_to_vertex[self.uv_faces.reshape(-1)] = self.faces.reshape(-1)
        cache = ROOT / "models" / f"anny_uv_{size}_v2.npz"
        if cache.exists():
            data = np.load(cache)
            self.face_id, self.bary = data["face_id"], data["bary"]
            self.parents1, self.parents2, self.fine_faces = data["parents1"], data["parents2"], data["fine_faces"]
        else:
            self.face_id, self.bary = self._rasterize()
            faces1, self.parents1 = _midpoint_subdivide(self.uv_faces, len(self.uv))
            self.fine_faces, self.parents2 = _midpoint_subdivide(faces1, len(self.parents1))
            cache.parent.mkdir(exist_ok=True)
            np.savez_compressed(cache, face_id=self.face_id, bary=self.bary, parents1=self.parents1,
                                parents2=self.parents2, fine_faces=self.fine_faces)
        self.fine_uv = self.refine(self.uv)
        self.covered = self.face_id >= 0
        codes = np.array([PARTS.index(p) for p in anny_body.vertex_part])
        self.texel_part = codes[self.faces[np.maximum(self.face_id, 0), 0]]
        # nearest covered texel for every texel: used to pad colour across seams
        _, labels = cv2.distanceTransformWithLabels((~self.covered).astype(np.uint8), cv2.DIST_L2, 5,
                                                    labelType=cv2.DIST_LABEL_PIXEL)
        ys, xs = np.nonzero(self.covered)
        lookup = np.zeros(labels.max() + 1, dtype=np.int64)
        lookup[labels[ys, xs]] = np.arange(len(ys))
        self.nearest = (ys[lookup[labels]], xs[lookup[labels]])

    def _rasterize(self):
        """For every texel: which triangle covers it and its barycentric weights."""
        n = self.size
        face_id = np.full((n, n), -1, dtype=np.int32)
        bary = np.zeros((n, n, 3), dtype=np.float32)
        px = self.uv[:, 0] * (n - 1)
        py = (1.0 - self.uv[:, 1]) * (n - 1)   # image row 0 is the top of the texture
        for f, (a, b, c) in enumerate(self.uv_faces):
            xs = np.array([px[a], px[b], px[c]]); ys = np.array([py[a], py[b], py[c]])
            x0, x1 = int(np.floor(xs.min())), int(np.ceil(xs.max()))
            y0, y1 = int(np.floor(ys.min())), int(np.ceil(ys.max()))
            d = (ys[1] - ys[2]) * (xs[0] - xs[2]) + (xs[2] - xs[1]) * (ys[0] - ys[2])
            if abs(d) < 1e-12:
                continue
            gx, gy = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
            w0 = ((ys[1] - ys[2]) * (gx - xs[2]) + (xs[2] - xs[1]) * (gy - ys[2])) / d
            w1 = ((ys[2] - ys[0]) * (gx - xs[2]) + (xs[0] - xs[2]) * (gy - ys[2])) / d
            w2 = 1 - w0 - w1
            inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
            face_id[gy[inside], gx[inside]] = f
            bary[gy[inside], gx[inside]] = np.stack([w0, w1, w2], axis=-1)[inside]
        return face_id, bary

    def spread(self, per_vertex: np.ndarray) -> np.ndarray:
        """Per-mesh-vertex values (V, C) -> texture image (size, size, C)."""
        per_vertex = np.asarray(per_vertex, dtype=np.float32)
        if per_vertex.ndim == 1:
            per_vertex = per_vertex[:, None]
        corners = per_vertex[self.faces[np.maximum(self.face_id, 0)]]            # (n, n, 3, C)
        return (corners * self.bary[..., None]).sum(axis=2)

    def pad(self, image: np.ndarray) -> np.ndarray:
        """Copy values outwards from the painted islands so seams never show the gaps."""
        return image[self.nearest]

    def refine(self, per_uv_vertex: np.ndarray) -> np.ndarray:
        """Values on the expanded mesh (U, C) -> values on the twice-split mesh."""
        return per_uv_vertex[self.parents1].mean(axis=1)[self.parents2].mean(axis=1)

    def sample(self, image: np.ndarray, uv: np.ndarray) -> np.ndarray:
        """Bilinear lookup of a texture-space image at UV points."""
        n, count, width = self.size, len(uv), 4096      # OpenCV limits one row to 32767 points
        pad = (-count) % width
        x = np.pad(uv[:, 0] * (n - 1), (0, pad)).astype(np.float32).reshape(-1, width)
        y = np.pad((1 - uv[:, 1]) * (n - 1), (0, pad)).astype(np.float32).reshape(-1, width)
        out = cv2.remap(image, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return out.reshape((-1,) + out.shape[2:])[:count]


# ------------------------------------------------------------------- relief

def _groove(distance, sigma):
    return np.exp(-(distance / sigma) ** 2)


def _to_segment(p, a, b):
    """Distance from 2D points p (..., 2) to the segment a-b."""
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    ab = b - a
    t = np.clip(((p - a) @ ab) / (ab @ ab), 0, 1)
    return np.linalg.norm(p - (a + t[..., None] * ab), axis=-1)


def _limb_frame(pos, start, end, side_sign):
    """Coordinates around a limb bone. Returns (s, theta, length): s runs 0..1 from start
    to end joint; theta is the angle around the bone in radians, 0 = front of the body,
    positive towards the outside of the body."""
    axis = end - start
    length = float(np.linalg.norm(axis))
    axis = axis / length
    front = np.array([0.0, -1.0, 0.0])
    front = front - (front @ axis) * axis
    front /= np.linalg.norm(front)
    lateral = np.cross(axis, front)
    if lateral[0] * side_sign < 0:
        lateral = -lateral
    rel = pos - start
    s = (rel @ axis) / length
    radial = rel - (rel @ axis)[..., None] * axis
    return s, np.arctan2(radial @ lateral, radial @ front), length


def anatomy_relief(anny_body: AnnyBody, layout: UVLayout) -> np.ndarray:
    """Height field over Anny's surface in UV space, in metres at full definition:
    positive = muscle belly standing out, negative = groove between muscles.

    Built on a reference body and stored per texel, so it follows the same anatomical
    spot on any body shape. All edges are Gaussian, a few millimetres wide: there are no
    hard lines to begin with.
    """
    H_CM = 175.0
    with torch.no_grad():
        body = anny_body.build("male", 30, H_CM, default_dials())
        lm = anny_body.landmarks(body)
    v = body["vertices"].numpy()
    joint = {k: j.numpy() for k, j in body["joints"].items()}
    H = H_CM / 100.0
    k = H / 1.75
    normals = np.asarray(trimesh.Trimesh(v, layout.faces, process=False).vertex_normals)
    pos = layout.spread(v)
    front = layout.spread(-normals[:, 1])[..., 0]
    x, z = pos[..., 0], pos[..., 2]
    ax = np.abs(x)
    part = layout.texel_part
    is_part = lambda name: part == PARTS.index(name)
    mm = 0.001
    h = np.zeros_like(x)

    # ---- front of the torso
    torso = is_part("torso")
    facing = np.clip((front - 0.15) / 0.35, 0, 1) * torso
    navel = float(v[anny_body.navel_vertex, 2])
    top = lm["chest"] - 0.050 * H                    # where the abdominal wall meets the ribs
    bottom = navel - 0.105 * H                       # pubis
    t = np.clip((z - bottom) / (top - bottom), 0, 1)
    half = (0.036 + 0.036 * t) * k                   # abdominal wall widens towards the ribs

    # rectus abdominis: four soft blocks a side, the left column sitting slightly higher
    # than the right, as it usually does
    levels = [bottom + 0.012, navel - 0.004, navel + (top - navel) * 0.36, navel + (top - navel) * 0.70, top]
    stagger = np.where(x > 0, 0.004, -0.002)
    blocks = np.zeros_like(x)
    for lo, hi in zip(levels[:-1], levels[1:]):
        centre_z, half_h = (lo + hi) / 2 + stagger, (hi - lo) / 2
        across = np.clip(1 - ((ax - half * 0.52) / (half * 0.50)) ** 2, 0, 1)
        along = np.clip(1 - ((z - centre_z) / half_h) ** 2, 0, 1)
        blocks = np.maximum(blocks, np.sqrt(across * along))
    wall = np.clip((half + 0.012 - ax) / 0.012, 0, 1) * np.clip((z - bottom) / 0.02, 0, 1) \
        * np.clip((top + 0.010 - z) / 0.015, 0, 1)
    h += (2.6 * blocks - 1.3) * mm * wall * facing
    h -= 1.4 * mm * _groove(x, 0.006 * k) * np.clip((z - bottom) / 0.03, 0, 1) \
        * np.clip((top + 0.03 - z) / 0.02, 0, 1) * facing                          # linea alba
    h -= 1.5 * mm * _groove(ax - half - 0.004, 0.008 * k) * wall * facing          # side lines

    # obliques above the hip, and the inguinal crease running down to the pubis
    flank = np.clip((front + 0.35) / 0.45, 0, 1) * torso
    h += 1.4 * mm * _groove(z - (navel + 0.015), 0.045) * _groove(ax - (half + 0.045 * k), 0.022 * k) * flank
    crease = _to_segment(np.stack([ax, z], -1), (0.100 * k, navel - 0.030 * H), (0.022 * k, bottom - 0.005))
    h -= 1.6 * mm * _groove(crease, 0.007 * k) * facing

    # serratus: four finger-like slips on the side of the ribs, pointing down and in
    for i in range(4):
        cx, cz = half + (0.050 - 0.004 * i) * k, top + 0.012 - 0.026 * i
        along = (ax - cx) * -0.72 + (z - cz) * -0.69
        across = (ax - cx) * 0.69 + (z - cz) * -0.72
        h += 1.5 * mm * np.exp(-(along / (0.020 * k)) ** 2 - (across / 0.0075) ** 2) * flank

    # pectoralis major: lower border, a little fullness above it, and the sternum between
    border = lm["chest"] - 0.042 * H + np.where(ax > 0.06 * k, 4.0 * (ax - 0.06 * k) ** 2,
                                                 1.5 * (0.06 * k - ax) ** 2)
    on_pec = np.clip((ax - 0.010) / 0.010, 0, 1) * np.clip((0.155 * k - ax) / 0.02, 0, 1) * facing
    h -= 1.8 * mm * _groove(z - border, 0.007) * on_pec
    h += 1.2 * mm * _groove(z - (border + 0.045), 0.035) * _groove(ax - 0.075 * k, 0.055 * k) * facing
    sternum_top = lm["chest"] + 0.085 * H
    h -= 1.5 * mm * _groove(x, 0.007 * k) * np.clip((z - (top + 0.02)) / 0.02, 0, 1) \
        * np.clip((sternum_top - z) / 0.03, 0, 1) * facing

    # ---- limbs, both sides
    for side, sign in ((".L", 1.0), (".R", -1.0)):
        mine = (x * sign > 0)
        # deltoid: its front and back borders meeting on the outside of the arm, a
        # little fullness between them, and the groove that continues down to the elbow
        s, th, length = _limb_frame(pos, joint["upperarm01" + side], joint["lowerarm01" + side], sign)
        coords = np.stack([s * length, 0.05 * th], -1)
        arm = is_part("upperarm") * mine
        deg = np.pi / 180
        borders = np.minimum(_to_segment(coords, (0.05 * length, 0.05 * 20 * deg), (0.47 * length, 0.05 * 82 * deg)),
                             _to_segment(coords, (0.05 * length, 0.05 * 160 * deg), (0.47 * length, 0.05 * 98 * deg)))
        h -= 1.6 * mm * _groove(borders, 0.006) * arm
        h += 1.2 * mm * np.exp(-((s - 0.22) / 0.16) ** 2 - ((th - 88 * deg) / (38 * deg)) ** 2) * arm
        h -= 1.0 * mm * _groove(_to_segment(coords, (0.50 * length, 0.05 * 90 * deg), (0.90 * length, 0.05 * 90 * deg)), 0.006) * arm

        # quadriceps: the two grooves either side of the middle muscle, the teardrop
        # above the inside of the knee, and the dip of the tendon above the kneecap
        s, th, length = _limb_frame(pos, joint["upperleg01" + side], joint["lowerleg01" + side], sign)
        coords = np.stack([s * length, 0.075 * th], -1)
        leg = is_part("upperleg") * mine * np.clip((s - 0.18) / 0.10, 0, 1)
        h -= 1.5 * mm * _groove(_to_segment(coords, (0.30 * length, 0.075 * 26 * deg), (0.86 * length, 0.075 * 20 * deg)), 0.007) * leg
        h -= 1.3 * mm * _groove(_to_segment(coords, (0.48 * length, 0.075 * -20 * deg), (0.86 * length, 0.075 * -12 * deg)), 0.007) * leg
        h += 1.5 * mm * np.exp(-((s - 0.84) / 0.10) ** 2 - ((th + 42 * deg) / (26 * deg)) ** 2) * leg
        h += 1.0 * mm * np.exp(-((s - 0.60) / 0.22) ** 2 - ((th - 48 * deg) / (28 * deg)) ** 2) * leg
        h -= 1.0 * mm * np.exp(-((s - 0.95) / 0.05) ** 2 - (th / (22 * deg)) ** 2) * leg

    h = np.clip(h, -0.9 * RELIEF_RANGE_M, 0.9 * RELIEF_RANGE_M).astype(np.float32)
    h[~layout.covered] = 0
    return layout.pad(h), layout.pad(pos)


def normal_map(height_m: np.ndarray, position_map: np.ndarray) -> np.ndarray:
    """Tangent-space normal map (RGB 0..1) for a height field in metres. Slopes are real
    slopes: height difference over surface distance, using how many metres one texel
    spans at each place. They are stored divided by NORMAL_SCALE_FULL, so the viewer's
    normalScale of NORMAL_SCALE_FULL x relief reproduces them."""
    per_col = np.clip(np.linalg.norm(np.gradient(position_map, axis=1), axis=2), 2e-4, 1e-2)
    per_row = np.clip(np.linalg.norm(np.gradient(position_map, axis=0), axis=2), 2e-4, 1e-2)
    slope_u = cv2.GaussianBlur(np.gradient(height_m, axis=1) / per_col, (0, 0), 1.0)
    slope_row = cv2.GaussianBlur(np.gradient(height_m, axis=0) / per_row, (0, 0), 1.0)
    gain = 1.0 / NORMAL_SCALE_FULL
    n = np.stack([-slope_u * gain, slope_row * gain, np.ones_like(slope_u)], axis=-1)   # +v is up the image
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return (n * 0.5 + 0.5).astype(np.float32)


# --------------------------------------------------------------- appearance

class Appearance:
    """The muscle relief maps the viewer draws on the clay body. One per server."""

    def __init__(self, anny_body: AnnyBody, size: int = SIZE):
        self.layout = UVLayout(anny_body, size)
        self.height_m, position_map = anatomy_relief(anny_body, self.layout)
        self.normal = normal_map(self.height_m, position_map)
        # where relief exists at all (for fading a photo's own definition)
        amount = cv2.GaussianBlur(np.abs(self.height_m), (0, 0), size * 0.008)
        self.relief_region = np.clip(amount / max(float(amount.max()), 1e-9) * 2.5, 0, 1).astype(np.float32)
        self.fine_height_m = self.layout.sample(self.height_m, self.layout.fine_uv)

    def displacement_png(self) -> bytes:
        img = np.clip(self.height_m / RELIEF_RANGE_M * 0.5 + 0.5, 0, 1)
        return cv2.imencode(".png", (img * 255).round().astype(np.uint8))[1].tobytes()

    def normal_png(self) -> bytes:
        return cv2.imencode(".png", cv2.cvtColor((self.normal * 255).round().astype(np.uint8),
                                                 cv2.COLOR_RGB2BGR))[1].tobytes()

    def fine_mesh(self, vertices: np.ndarray, relief: float):
        """The mesh as the viewer draws it: split twice and displaced. Returns
        (positions, faces). Used for checks and exports."""
        lay = self.layout
        mesh = trimesh.Trimesh(vertices, lay.faces, process=False)
        normals = lay.refine(np.asarray(mesh.vertex_normals)[lay.uv_to_vertex])
        normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
        return lay.refine(vertices[lay.uv_to_vertex]) + normals * (self.fine_height_m * relief)[:, None], lay.fine_faces


# ------------------------------------------------------------ photoreal 2D view

def background_plate(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The photo with the person removed (LaMa). The model (196 MB, downloaded once into
    models/torch) is loaded for this call and released afterwards."""
    import os
    os.environ.setdefault("TORCH_HOME", str(ROOT / "models" / "torch"))
    from PIL import Image
    from simple_lama_inpainting import SimpleLama

    h, w = mask.shape
    grow = max(6, int(0.02 * max(h, w)))   # also covers soft edges and contact shadow near the body
    hole = cv2.dilate(mask.astype(np.uint8) * 255, np.ones((3, 3), np.uint8), iterations=grow)
    out = SimpleLama()(Image.fromarray(image_rgb), Image.fromarray(hole))
    return np.asarray(out)[:h, :w].copy()


def _rasterize(pixel_xy, depth, faces, size):
    """Which triangle is in front at each pixel, and where inside it. Camera view."""
    w, h = size
    face_id = np.full((h, w), -1, dtype=np.int32)
    bary = np.zeros((h, w, 3), dtype=np.float32)
    zbuf = np.full((h, w), np.inf, dtype=np.float32)
    px, py = pixel_xy[:, 0], pixel_xy[:, 1]
    for f, (a, b, c) in enumerate(faces):
        xs = np.array([px[a], px[b], px[c]]); ys = np.array([py[a], py[b], py[c]])
        d = (ys[1] - ys[2]) * (xs[0] - xs[2]) + (xs[2] - xs[1]) * (ys[0] - ys[2])
        if d >= -1e-9:          # facing away from the camera (or edge-on)
            continue
        x0, x1 = max(int(np.floor(xs.min())), 0), min(int(np.ceil(xs.max())), w - 1)
        y0, y1 = max(int(np.floor(ys.min())), 0), min(int(np.ceil(ys.max())), h - 1)
        if x0 > x1 or y0 > y1:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        w0 = ((ys[1] - ys[2]) * (gx - xs[2]) + (xs[2] - xs[1]) * (gy - ys[2])) / d
        w1 = ((ys[2] - ys[0]) * (gx - xs[2]) + (xs[0] - xs[2]) * (gy - ys[2])) / d
        w2 = 1 - w0 - w1
        z = w0 * depth[a] + w1 * depth[b] + w2 * depth[c]
        sub = zbuf[y0:y1 + 1, x0:x1 + 1]
        hit = (w0 >= 0) & (w1 >= 0) & (w2 >= 0) & (z < sub)
        if hit.any():
            sub[hit] = z[hit]
            face_id[y0:y1 + 1, x0:x1 + 1][hit] = f
            bary[y0:y1 + 1, x0:x1 + 1][hit] = np.stack([w0, w1, w2], axis=-1)[hit]
    return face_id, bary


def _head_registration(vertices, landmark_groups, keypoints):
    """Scale, rotation and shift (s, R, t) that carry the model's eyes, nose and ears
    onto the same landmarks of the SAM body (both in the neutral pose): x -> s R x + t.
    Ears count half: the two models mark them at slightly different points."""
    weights = {"left_eye": 1.0, "right_eye": 1.0, "nose": 1.0, "left_ear": 0.5, "right_ear": 0.5}
    names = [n for n in weights if n in keypoints]
    if len(names) < 4:
        return None
    A = np.stack([vertices[landmark_groups[n]].mean(axis=0) for n in names])
    B = np.stack([np.asarray(keypoints[n], dtype=np.float64) for n in names])
    w = np.array([weights[n] for n in names]); w = w / w.sum()
    ca, cb = (w[:, None] * A).sum(0), (w[:, None] * B).sum(0)
    A0, B0 = A - ca, B - cb
    U, D, Vt = np.linalg.svd((w[:, None] * B0).T @ A0)
    sign = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ sign @ Vt
    s = float((D * np.diag(sign)).sum() / (w * (A0 ** 2).sum(1)).sum())
    s = min(max(s, 0.85), 1.25)        # a head is a head: never trust a wild scale
    return s, R, cb - s * R @ ca


def _fill_from(image, known, hole, scale):
    """Colour for the `hole` pixels taken only from nearby `known` pixels: averages over
    shrinking neighbourhoods, the finest one with enough known pixels winning. Smooth,
    and never picks up colour from anything outside `known`."""
    img = image.astype(np.float32)
    k = known.astype(np.float32)
    fill = np.zeros_like(img)
    for i, sigma in enumerate((16 * scale, 8 * scale, 4 * scale, 2 * scale, scale)):
        den = cv2.GaussianBlur(k, (0, 0), sigma)
        est = cv2.GaussianBlur(img * k[..., None], (0, 0), sigma) / np.maximum(den, 1e-6)[..., None]
        support = (den > 1e-4).astype(np.float32) if i == 0 else np.clip(den / 0.15, 0, 1)
        fill = est * support[..., None] + fill * (1 - support[..., None])
    out = img.copy()
    out[hole] = fill[hole]
    return out


def _affine(src, dst):
    """3x3 matrix A with A @ [x, y, 1] ~ the matching dst point (least squares)."""
    A = np.linalg.lstsq(np.hstack([src, np.ones((len(src), 1))]), dst, rcond=None)[0].T
    return np.vstack([A, [0.0, 0.0, 1.0]])


def _rotation(pivot, angle):
    """3x3 matrix turning image points by angle (radians) about pivot."""
    c, s = np.cos(angle), np.sin(angle)
    px, py = pivot
    return np.array([[c, -s, px - c * px + s * py], [s, c, py - s * px - c * py], [0.0, 0.0, 1.0]])


class PhotoView:
    """The original photo, re-drawn for a morphed body, from the photo's own camera.

    Nothing is generated: every pixel of the person is a pixel of the real photo, moved.
    The person is split into layers by the fitted mesh, and each layer moves on its own:

      background  the photo with the person removed (LaMa), made once.
      body        trunk, legs and head. The morphed mesh (without its arms) is drawn in
                  the camera view; each of its pixels knows which point of the skin it
                  shows and that point's place in the photo. Pixels the mesh does not
                  cover (hair, loose cloth) follow the nearest part of the body, fading
                  with distance. Body the photo only saw behind an arm is filled in from
                  the body around it, so it can show when the arm moves away.
      arms        each arm is moved whole, by the straight-line (affine) map that best
                  carries its mesh vertices to their new places: it shifts, turns and
                  widens with the mesh but is never squashed by the trunk. When the
                  trunk widens into an arm, the arm swings out from the shoulder.
      head        the original pixels, never moved, so the face stays the person's own.

    Drawn in that order with soft (about 2 px) edges. The body is shaded for its change
    of shape with lighting estimated from the photo itself (fuller sides turn a little
    darker, a slimmer waist shows its contour), and the muscle relief for the new body
    fat is lit the same way.
    """

    def __init__(self, image_rgb, mask, skin, photo_pose, anny_body: AnnyBody,
                 start_posed_vertices, appearance: Appearance, keypoints=None):
        self.image, self.mask, self.skin = image_rgb, mask, skin
        self.pp, self.app = photo_pose, appearance
        self.faces = anny_body.faces.numpy()
        # Only the body is fitted, so the head can sit centimetres away from where SAM
        # saw it. Line it up on eyes, nose and ears, or the face is drawn off its place.
        self.head_share = anny_body.head_share[:, None]
        self.head_fix = None
        if keypoints is not None:
            self.head_fix = _head_registration(start_posed_vertices, anny_body.face_landmarks, keypoints)
        self.face_is_head = (anny_body.vertex_part[self.faces] == "head").all(axis=1) | \
                            (anny_body.vertex_part[self.faces] == "neck").all(axis=1)
        # Region of every triangle: 0 trunk and head, 1/2 left/right arm, 3/4 left/right leg.
        first = self.faces[:, 0]
        limb = {"upperarm": 1, "lowerarm": 1, "hand": 1, "upperleg": 3, "lowerleg": 3, "foot": 3}
        region = np.array([limb.get(p, 0) for p in anny_body.vertex_part[first]])
        self.face_region = region + ((region > 0) & ~anny_body.vertex_left[first])
        on_arm = np.isin(self.face_region, (1, 2))
        self.body_faces, self.arm_faces = np.nonzero(~on_arm)[0], np.nonzero(on_arm)[0]
        # Each arm's vertices, and the ring where it joins the trunk (its pivot).
        self.arm_vertices, self.arm_seam = {}, {}
        for k in (1, 2):
            mine = np.unique(self.faces[self.face_region == k])
            self.arm_vertices[k] = mine
            self.arm_seam[k] = np.intersect1d(mine, np.unique(self.faces[self.face_region != k]))
        w, h = self.size = (image_rgb.shape[1], image_rgb.shape[0])
        self.start_xy, self.facing, cam, self.start_normals, self.match = self._carry(start_posed_vertices, None)
        face_id, bary = _rasterize(self.start_xy, cam[:, 2], self.faces, self.size)
        # Which region every pixel of the photo belongs to: that of the nearest body pixel.
        _, near = self._nearest(face_id >= 0)
        owner = self.face_region[np.maximum(face_id, 0)][near]
        self.start_head = self._head(face_id)
        self.light = self._estimate_light(face_id, bary, self.start_normals)
        self.plate = background_plate(image_rgb, mask)

        # ---- the layers, in the photo
        person = mask.astype(bool)
        arm_px = person & np.isin(owner, (1, 2))
        body_fid, _ = self._raster(self.start_xy, cam, self.body_faces)
        # Body the photo saw only behind an arm: arm pixels with trunk or leg mesh under
        # them, arm pixels with body on both sides in the same row (a forearm hanging in
        # front of the hip), and a strip of arm along every edge it shares with the body,
        # so the arm's soft edge always lies over body, never over the background.
        body_px = person & ~arm_px
        strip = max(6, int(round(0.025 * max(w, h))))
        along = cv2.dilate(body_px.astype(np.uint8), cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * strip + 1, 2 * strip + 1))) > 0
        span = max(9, int(round(0.08 * max(w, h))))
        row = np.ones((1, span), np.uint8)
        body_right = cv2.dilate(body_px.astype(np.uint8), row, anchor=(0, 0)) > 0         # body within span to the right
        body_left = cv2.dilate(body_px.astype(np.uint8), row, anchor=(span - 1, 0)) > 0   # and to the left
        mesh_under = cv2.dilate(((body_fid >= 0) & arm_px).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        crossing = arm_px & body_left & body_right
        behind = arm_px & (mesh_under | along | crossing)
        # Filled-in body only counts near the new body mesh, or it would show as a ghost
        # of the old outline where an arm moves away.
        self.fill_needs_mesh = behind.astype(np.float32)
        dist_body, nearest_body = self._nearest(body_px)
        self.body_label = owner.copy()                      # region each body-layer pixel belongs to
        self.body_label[behind] = np.where(body_fid >= 0, self.face_region[np.maximum(body_fid, 0)],
                                           owner[nearest_body])[behind]
        self.body_alpha = (body_px | behind).astype(np.float32)
        # Fill colour comes from inside the body only (outline pixels are mixed with the
        # background), and the fill starts from the real pixels at the arm's edge.
        inside = cv2.erode(person.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=4) > 0
        known = (cv2.erode(body_px.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) & inside
        filled = _fill_from(image_rgb, known, behind, scale=max(1.0, 0.004 * max(w, h)))
        ramp = np.clip((dist_body - 1.0) / 4.0, 0, 1)[..., None] * behind[..., None]
        self.body_image = image_rgb.astype(np.float32) * (1 - ramp) + filled * ramp
        self.arm_alpha = {k: (person & (owner == k)).astype(np.float32) for k in (1, 2)}
        # How far, row by row, the trunk and leg mesh reaches behind each arm in the photo.
        arm_fid, _ = self._raster(self.start_xy, cam, self.arm_faces)
        self.overlap = {k: ((body_fid >= 0) & (self.face_region[np.maximum(arm_fid, 0)] == k) & (arm_fid >= 0))
                        .sum(axis=1).astype(np.float32) for k in (1, 2)}
        # Where the head is fully locked, its original pixels are drawn last, over everything.
        self.head_alpha = np.clip((self.start_head * 1.5 - 0.8) / 0.2, 0, 1) * person

    def _carry(self, anny_vertices, match):
        """Anny vertices (fit pose) -> photo pixels, how much each faces the camera,
        camera-space points and normals. The first pairing with the SAM body is reused,
        so the same skin point is followed from the fitted body to a morphed one."""
        if self.head_fix is not None:
            s, R, t = self.head_fix
            anny_vertices = anny_vertices + self.head_share * (s * anny_vertices @ R.T + t - anny_vertices)
        normals = np.asarray(trimesh.Trimesh(anny_vertices, self.faces, process=False).vertex_normals)
        N = self.pp.N.numpy()
        if match is None:
            mhr_normals = np.asarray(trimesh.Trimesh(N, self.pp.faces, process=False).vertex_normals)
            _, match = cKDTree(np.hstack([N, 0.05 * mhr_normals])).query(
                np.hstack([anny_vertices, 0.05 * normals]))
        R = self.pp.R.numpy()[match]
        cam = self.pp.P.numpy()[match] + (R @ ((anny_vertices - N[match]) / self.pp.scale)[:, :, None])[:, :, 0]
        w, h = self.size
        f = self.pp.view["focal"]
        xy = np.stack([f * cam[:, 0] / cam[:, 2] + w / 2, f * cam[:, 1] / cam[:, 2] + h / 2], axis=1)
        cam_normals = (R @ normals[:, :, None])[:, :, 0]
        toward = -(cam_normals * (cam / np.linalg.norm(cam, axis=1, keepdims=True))).sum(1)
        return xy.astype(np.float32), np.clip(toward, 0, 1).astype(np.float32), cam, cam_normals, match

    def _raster(self, xy, cam, subset):
        """_rasterize over some of the triangles; ids are of the whole mesh."""
        fid, bary = _rasterize(xy, cam[:, 2], self.faces[subset], self.size)
        return np.where(fid >= 0, subset[np.maximum(fid, 0)], -1), bary

    def _interp(self, per_vertex, fid, bary):
        return (per_vertex[self.faces[np.maximum(fid, 0)]] * bary[..., None]).sum(axis=2)

    @staticmethod
    def _nearest(covered):
        """For every pixel: distance to the nearest covered pixel, and that pixel's (row, column)."""
        dist, labels = cv2.distanceTransformWithLabels((~covered).astype(np.uint8), cv2.DIST_L2, 5,
                                                       labelType=cv2.DIST_LABEL_PIXEL)
        ys, xs = np.nonzero(covered)
        lookup = np.zeros(labels.max() + 1, dtype=np.int64)
        lookup[labels[ys, xs]] = np.arange(len(ys))
        return dist, (ys[lookup[labels]], xs[lookup[labels]])

    def _head(self, face_id):
        head = (face_id >= 0) & self.face_is_head[np.maximum(face_id, 0)]
        # Wide fall-off: where the trapezius moves and the neck does not, the outline
        # bends gently instead of stepping.
        return cv2.GaussianBlur(head.astype(np.float32), (0, 0), max(2.0, 0.025 * max(self.size)))

    def _estimate_light(self, face_id, bary, normals):
        """Lighting of the photo as brightness = c0 + c . normal, fitted on bare skin.
        Falls back to soft light from above and in front when there is too little skin."""
        covered = face_id >= 0
        n = self._interp(normals, face_id, bary)
        core = cv2.erode(self.mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=ERODE_PX) > 0
        use = covered & core & (self.skin if self.skin is not None else core)
        fallback = np.array([0.6, 0.0, -0.25, -0.35], dtype=np.float32)
        if use.sum() < 2000:
            return fallback
        lum = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY).astype(np.float32)[use] / 255.0
        lum = lum / max(float(np.median(lum)), 1e-3) * 0.6
        A = np.concatenate([np.ones((use.sum(), 1), np.float32), n[use]], axis=1)
        coef = np.linalg.lstsq(A, lum, rcond=None)[0].astype(np.float32)
        return coef if np.isfinite(coef).all() and coef[0] > 0.1 else fallback

    def _brightness(self, n):
        return (self.light[0] + n @ self.light[1:]).clip(0.05)

    def _relief(self, layer, skin, fid, bary, normals, cam, start_relief, relief):
        """A layer re-lit for the new muscle relief, from the UV maps at its pixels.
        More defined than the photo: the extra relief is lit with the photo's own light.
        Less defined: the photo's own definition fades."""
        change = relief - start_relief
        covered = fid >= 0
        if abs(change) <= 1e-3 or not covered.any():
            return layer
        w, h = self.size
        layout = self.app.layout
        n_tex = layout.size
        uv = (layout.uv[layout.uv_faces[np.maximum(fid, 0)]] * bary[..., None]).sum(axis=2)
        tu = (uv[..., 0] * (n_tex - 1)).astype(np.float32)
        tv = ((1 - uv[..., 1]) * (n_tex - 1)).astype(np.float32)
        if change > 0:
            normal = self._interp(normals, fid, bary)
            normal /= np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-9
            metres_per_px = float(np.median(cam[:, 2])) / self.pp.view["focal"]
            height = cv2.GaussianBlur(cv2.remap(self.app.height_m, tu, tv, cv2.INTER_LINEAR) * covered, (0, 0), 1.0)
            slope_y, slope_x = np.gradient(height * change / metres_per_px)
            bumped = normal - np.stack([slope_x, slope_y, np.zeros_like(slope_x)], axis=-1)
            bumped /= np.linalg.norm(bumped, axis=-1, keepdims=True) + 1e-9
            ratio = np.clip(self._brightness(bumped) / self._brightness(normal), 0.72, 1.28)
            ratio = 1 + (cv2.GaussianBlur(ratio.astype(np.float32), (0, 0), 0.7) - 1) * skin * covered
            return layer * ratio[..., None]
        region = cv2.remap(self.app.relief_region, tu, tv, cv2.INTER_LINEAR) * skin * covered
        k = cv2.GaussianBlur((min(1.0, -change * 1.6) * region).astype(np.float32), (0, 0), 2.0)[..., None]
        return layer * (1 - k) + cv2.GaussianBlur(layer, (0, 0), 0.012 * max(w, h)) * k

    def render(self, posed_vertices, start_relief: float, relief: float, passes: bool = False):
        """posed_vertices: the morphed Anny body in the fit pose (same frame as the
        vertices given at construction). Returns the re-drawn photo (RGB), or with
        passes=True a dict that also holds the depth, normal, silhouette, albedo and
        head lock passes."""
        w, h = self.size
        xy, _, cam, normals, _ = self._carry(posed_vertices, self.match)
        gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        photo = self.image.astype(np.float32)
        skin_src = np.ones((h, w), np.float32) if self.skin is None else self.skin.astype(np.float32)
        feather = lambda a: cv2.GaussianBlur(a.astype(np.float32), (0, 0), 1.0)

        # ---- body layer: the mesh without its arms tells every pixel where it comes from
        fid_b, bary_b = self._raster(xy, cam, self.body_faces)
        covered = fid_b >= 0
        source = self._interp(self.start_xy, fid_b, bary_b)
        move = np.where(covered[..., None], source - np.stack([gx, gy], axis=-1), 0).astype(np.float32)
        dist, near = self._nearest(covered)
        fade = np.exp(-dist / (0.04 * max(w, h)))[..., None]
        # Each pixel beyond the mesh copies its nearest mesh pixel; smoothed, so the
        # outline (top of the shoulders, hair) stays smooth instead of ragged.
        smooth = lambda f: cv2.GaussianBlur(f.astype(np.float32), (0, 0), 0.012 * max(w, h))
        move = np.where(covered[..., None], move, smooth(move[near] * fade))
        # A leg only carries pixels that were its own (or the trunk's): in the gap between
        # the legs one leg's movement must not reach across and drag the other.
        giver = self.face_region[np.maximum(fid_b, 0)][near]
        sx = np.clip(np.rint(gx + move[..., 0]), 0, w - 1).astype(np.int32)
        sy = np.clip(np.rint(gy + move[..., 1]), 0, h - 1).astype(np.int32)
        label = self.body_label[sy, sx]
        own = covered | (label == giver) | (label == 0) | (giver == 0)
        move = move * own[..., None]
        # Identity lock: the head keeps its original pixels.
        lock = np.clip(np.maximum(self.start_head, self._head(fid_b)) * 1.5, 0, 1)
        move = cv2.GaussianBlur(move * (1 - lock[..., None]), (0, 0), 1.5)
        mx, my = gx + move[..., 0], gy + move[..., 1]
        # Edges of the photo continue past the frame (a cropped body goes on below it).
        look = lambda a: cv2.remap(a, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        body_albedo = look(self.body_image)
        # (the fitted mesh sits a little inside the real outline, so allow a margin)
        stray = (look(self.fill_needs_mesh) > 0.5) & (dist > 0.02 * max(w, h))
        body_alpha = feather(look(self.body_alpha) * own * ~stray)
        body_skin = look(skin_src) * covered * (1 - lock)
        # Shading for the change of shape: brightness of the new surface direction over
        # that of the same skin point in the photo, with the photo's own light.
        n_new = self._interp(normals, fid_b, bary_b)
        n_new /= np.linalg.norm(n_new, axis=-1, keepdims=True) + 1e-9
        n_old = self._interp(self.start_normals, fid_b, bary_b)
        n_old /= np.linalg.norm(n_old, axis=-1, keepdims=True) + 1e-9
        shade = np.clip(self._brightness(n_new) / self._brightness(n_old), 0.82, 1.2)
        facing = np.clip((-n_new[..., 2] - 0.15) / 0.3, 0, 1)     # fades out towards the outline
        shade = 1 + (cv2.GaussianBlur(shade.astype(np.float32), (0, 0), 0.004 * max(w, h)) - 1) \
            * facing * covered * (1 - lock)
        body = self._relief(body_albedo * shade[..., None], body_skin, fid_b, bary_b, normals, cam,
                            start_relief, relief)

        # ---- arm layers: each moved whole
        fid_a, bary_a = self._raster(xy, cam, self.arm_faces)
        out = self.plate.astype(np.float32) * (1 - body_alpha[..., None]) + body * body_alpha[..., None]
        albedo = body_albedo * body_alpha[..., None]
        person_alpha = body_alpha.copy()
        clear = 1 - body_alpha                     # how much background still shows, layer by layer
        arms_solid = np.zeros((h, w), bool)
        rows = np.arange(h, dtype=np.float32)
        for k in (1, 2):
            idx = self.arm_vertices[k]
            if self.arm_alpha[k].sum() < 50:
                continue
            M = _affine(xy[idx], self.start_xy[idx])          # new place -> place in the photo
            # Does the trunk mesh now reach further behind the arm than in the photo (the
            # meshes pass through each other)? Then the arm swings out from the shoulder by
            # about that much. Only rows well below the armpit count: near the pivot a pixel
            # of wobble would read as a large angle.
            pivot = xy[self.arm_seam[k]].mean(axis=0)
            arm_mesh = (fid_a >= 0) & (self.face_region[np.maximum(fid_a, 0)] == k)
            extra = ((fid_b >= 0) & arm_mesh).sum(axis=1) - self.overlap[k]
            extra = np.convolve(extra.astype(np.float32), np.ones(5) / 5, mode="same")
            reach = rows - pivot[1]
            arm_rows = np.nonzero(arm_mesh.any(axis=1))[0]
            length = float(arm_rows.max() - pivot[1]) if len(arm_rows) else 0.0
            test = (reach > 0.35 * length) & arm_mesh.any(axis=1) & (extra > 2.0)
            grid = np.stack([gx, gy, np.ones_like(gx)], axis=-1)
            src = grid @ M[:2].T
            if length > 0 and test.sum() >= 3:
                angle = min(float(np.percentile(extra[test] / reach[test], 90)), np.radians(10))
                if angle > 1e-3:
                    side = 1.0 if xy[idx, 0].mean() > xy[:, 0].mean() else -1.0
                    # Turning by -side*angle moves the hanging arm away from the body. It
                    # fades in below the shoulder, so the top of the arm does not tip into
                    # the trapezius.
                    turned = grid @ (M @ np.linalg.inv(_rotation(pivot, -side * angle)))[:2].T
                    t = np.clip((gy - pivot[1]) / (0.3 * length), 0, 1)[..., None]
                    src = src * (1 - t) + turned * t
            # Where the arm joins the trunk it follows the mesh pixel by pixel, like the
            # trunk beside it, so the two meet without a crack; further down it moves whole.
            mine = (fid_a >= 0) & (self.face_region[np.maximum(fid_a, 0)] == k)
            if mine.any():
                flow = np.where(mine[..., None], self._interp(self.start_xy, fid_a, bary_a), 0)
                _, near_a = self._nearest(mine)
                grid2 = np.stack([gx, gy], -1)
                flow = np.where(mine[..., None], flow, smooth(flow[near_a] - grid2[near_a]) + grid2)
                seam = np.zeros((h, w), np.uint8)
                for x_, y_ in np.round(xy[self.arm_seam[k]]).astype(int):
                    if 0 <= x_ < w and 0 <= y_ < h:
                        seam[y_, x_] = 1
                d_seam = cv2.distanceTransform(1 - seam, cv2.DIST_L2, 5)
                near_seam = np.exp(-(d_seam / (0.04 * max(w, h))) ** 2)[..., None]
                src = src * (1 - near_seam) + flow * near_seam
            sx_a, sy_a = src[..., 0].astype(np.float32), src[..., 1].astype(np.float32)
            warp = lambda a: cv2.remap(a, sx_a, sy_a, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            arm_albedo = warp(photo)
            arm_alpha = feather(warp(self.arm_alpha[k]))
            arm_skin = warp(skin_src) * (fid_a >= 0)
            arm = self._relief(arm_albedo, arm_skin, fid_a, bary_a, normals, cam, start_relief, relief)
            out = out * (1 - arm_alpha[..., None]) + arm * arm_alpha[..., None]
            albedo = albedo * (1 - arm_alpha[..., None]) + arm_albedo * arm_alpha[..., None]
            person_alpha = np.maximum(person_alpha, arm_alpha)
            clear = clear * (1 - arm_alpha)
            arms_solid |= arm_alpha > 0.5

        # ---- head: the original pixels, last
        head = feather(self.head_alpha)[..., None]
        out = out * (1 - head) + photo * head
        out = np.clip(out, 0, 255).astype(np.uint8)
        # Slits: where layers that met in the photo (an arm lying against the trunk) drift
        # a little apart, close the slit from the person around it. Only where the photo
        # showed the person: background the photo already showed stays background, and
        # wider gaps stay open.
        clear = clear * (1 - head[..., 0])
        reach = max(3, int(round(0.01 * max(w, h))))
        grow = lambda a: cv2.dilate(a.astype(np.uint8), cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * reach + 1, 2 * reach + 1))) > 0
        crack = (clear > 0.05) & grow(body_alpha > 0.5) & grow(arms_solid) & self.mask.astype(bool)
        if crack.any():
            out = cv2.inpaint(out, crack.astype(np.uint8), 3, cv2.INPAINT_TELEA)
            person_alpha = np.maximum(person_alpha, crack.astype(np.float32))
        if not passes:
            return out
        # Depth and normals of whichever mesh part is in front at each pixel.
        depth_b = np.where(covered, self._interp(cam[:, 2:3], fid_b, bary_b)[..., 0], np.inf)
        depth_a = np.where(fid_a >= 0, self._interp(cam[:, 2:3], fid_a, bary_a)[..., 0], np.inf)
        front_arm = depth_a < depth_b
        n_arm = self._interp(normals, fid_a, bary_a)
        n_arm /= np.linalg.norm(n_arm, axis=-1, keepdims=True) + 1e-9
        normal = np.where(front_arm[..., None], n_arm, np.where(covered[..., None], n_new, 0))
        return {"image": out, "depth": np.minimum(depth_a, depth_b).astype(np.float32),
                "covered": covered | (fid_a >= 0), "normal": normal.astype(np.float32),
                "albedo": np.clip(albedo, 0, 255).astype(np.uint8), "alpha": person_alpha,
                "head_lock": np.maximum(lock, feather(self.head_alpha)).astype(np.float32)}
