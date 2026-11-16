"""Stage 5: appearance. Muscle relief, clean photo texture, and the photoreal 2D view.

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

2. Photo texture (photo builds)
   Every texel must pass a gate before it may take a pixel from the photo:
     - it projects inside the person cut-out eroded by 4 px,
     - at least 5 px from every image border (cropped feet, heads, hands),
     - it is not on a hand (gloves, watches, rings and posed fingers never bake in),
     - and, for the sharp photo, its skin faced the camera.
   Texels that fail the last test but are behind the person get a heavily blurred copy
   of the cut-out at the same spot (clothing keeps its colour on the back, no fake
   detail). Everything else is grown in from neighbouring data over a short distance
   and settles on the subject's own median skin tone, measured from texels that are
   bare skin. Transitions are feathered.

3. Photoreal 2D view (photo builds)
   The morphed body is drawn from the photo's own camera into depth, normal and albedo
   passes. The albedo pass is the real photo, moved pixel by pixel. New relief is lit
   with lighting estimated from the photo itself (brightness against surface direction
   on bare skin), the head is locked to the original pixels, and the result is laid
   over a background with the person removed (LaMa).

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
CLAY = (201, 164, 138)        # clay colour for bodies made from numbers
ERODE_PX = 4                  # cut-out is shrunk by this much before any pixel is sampled
BORDER_PX = 5                 # texels projecting this close to the image border are not sampled
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
    """Everything the viewer needs to draw a body's skin. One per server."""

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

    def clay(self) -> np.ndarray:
        """Plain clay colour. Relief is geometry and lighting, not colour."""
        return np.full((self.layout.size, self.layout.size, 3), CLAY, dtype=np.uint8)

    def bake_photo(self, image_rgb, mask, skin, pixel_xy, facing, fallback_tone=None):
        """Front photo -> texture in Anny's UV layout.

        pixel_xy (V, 2) and facing (V,) are per mesh vertex (PhotoView computes them).
        skin is the bare-skin mask from stage 1 (or None).
        Returns (texture uint8, skin_texels bool, skin_tone RGB).
        """
        lay = self.layout
        h, w = mask.shape
        core = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=ERODE_PX).astype(np.float32)
        img = image_rgb.astype(np.float32)
        sigma = max(h, w) * 0.02
        weight = cv2.GaussianBlur(core, (0, 0), sigma)        # blurred cut-out: no background in the blur
        soft = cv2.GaussianBlur(img * core[..., None], (0, 0), sigma) / np.maximum(weight, 1e-4)[..., None]

        px = lay.spread(pixel_xy)
        face = lay.spread(facing)[..., 0]
        mx, my = px[..., 0].astype(np.float32), px[..., 1].astype(np.float32)
        look = lambda a, mode=cv2.INTER_LINEAR: cv2.remap(a, mx, my, mode, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        # the gate
        in_bounds = (mx >= BORDER_PX) & (mx <= w - 1 - BORDER_PX) & (my >= BORDER_PX) & (my <= h - 1 - BORDER_PX)
        allowed = lay.covered & in_bounds & (lay.texel_part != PARTS.index("hand"))
        sharp_ok = allowed & (look(core) > 0.99)
        soft_ok = allowed & (look(weight) > 0.5)

        sharp = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        blurred = cv2.remap(soft, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        w_sharp = np.clip((face - 0.25) / 0.35, 0, 1) * sharp_ok
        w_soft = (1 - w_sharp) * soft_ok

        # the subject's own skin tone, from texels that are sharp, front-facing bare skin
        seen = sharp_ok & (face > 0.5)
        is_skin = seen & (look(skin.astype(np.float32)) > 0.5) if skin is not None else np.zeros_like(seen)
        if is_skin.sum() >= 300:
            tone = np.median(sharp[is_skin], axis=0)
        elif fallback_tone is not None:
            tone = np.asarray(fallback_tone, dtype=np.float32)
        elif seen.any():
            tone = np.median(sharp[seen], axis=0)
        else:
            tone = np.array(CLAY, dtype=np.float32)

        # Feather: a weight can only fall off towards a blocked area, never rise into it.
        feather = lambda a, ok: cv2.GaussianBlur(a.astype(np.float32), (0, 0), 3.0) * ok
        w_sharp, w_soft = feather(w_sharp, sharp_ok), feather(w_soft, soft_ok)
        known = np.clip(w_sharp + w_soft, 0, 1)
        data = (sharp * w_sharp[..., None] + blurred * w_soft[..., None]) / np.maximum(w_sharp + w_soft, 1e-4)[..., None]
        # Texels with no photo data: grow colour in from neighbouring data over a short
        # distance, settling on the skin tone further away. Short, so colour cannot
        # creep across the gaps between UV islands onto another body part.
        def grown(sigma, prior):
            return (cv2.GaussianBlur(data * known[..., None], (0, 0), sigma) + prior * tone[None, None]) /                    (cv2.GaussianBlur(known, (0, 0), sigma) + prior)[..., None]
        near = np.clip(cv2.GaussianBlur(known, (0, 0), 4.0) / 0.3, 0, 1)[..., None]
        fill = grown(12.0, 0.05) * (1 - near) + grown(4.0, 0.02) * near
        tex = data * known[..., None] + fill * (1 - known[..., None])
        tex = lay.pad(np.where(lay.covered[..., None], tex, 0))
        return np.clip(tex, 0, 255).astype(np.uint8), is_skin, tone

    def restyle(self, base: np.ndarray, skin_texels: np.ndarray, start_relief: float, relief: float) -> np.ndarray:
        """Photo texture for a morphed body. The colour map is never drawn on. Only when
        the body is LESS defined than in the photo is bare skin smoothed where muscle
        relief lives, so the photo's own definition fades with it."""
        if relief >= start_relief:
            return base
        tex = base.astype(np.float32)
        smooth = cv2.GaussianBlur(tex, (0, 0), self.layout.size * 0.012)
        area = cv2.GaussianBlur((skin_texels & self.layout.covered).astype(np.float32), (0, 0), 2.0)
        k = (min(1.0, (start_relief - relief) * 1.6) * self.relief_region * area)[..., None]
        return np.clip(tex * (1 - k) + smooth * k, 0, 255).astype(np.uint8)


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


class PhotoView:
    """The original photo, re-drawn for a morphed body, from the photo's own camera.

    Nothing is generated: every pixel of the person is a pixel of the real photo, moved.
    The morphed mesh is drawn in the camera view; each of its pixels knows which point
    of the skin it shows and that point's place in the original photo. Pixels of the
    person the mesh does not cover (hair, loose cloth, gloves) follow the nearest part
    of the body, fading out with distance. The head never moves, so the face stays the
    person's own. Where the body got smaller, the background plate shows through.
    """

    def __init__(self, image_rgb, mask, skin, photo_pose, anny_body: AnnyBody,
                 start_posed_vertices, appearance: Appearance):
        self.image, self.mask, self.skin = image_rgb, mask, skin
        self.pp, self.app = photo_pose, appearance
        self.faces = anny_body.faces.numpy()
        self.face_is_head = (anny_body.vertex_part[self.faces] == "head").all(axis=1) | \
                            (anny_body.vertex_part[self.faces] == "neck").all(axis=1)
        self.size = (image_rgb.shape[1], image_rgb.shape[0])
        self.start_xy, self.facing, cam, normals, self.match = self._carry(start_posed_vertices, None)
        face_id, bary = _rasterize(self.start_xy, cam[:, 2], self.faces, self.size)
        self.start_head = self._head(face_id)
        self.light = self._estimate_light(face_id, bary, normals)
        self.plate = background_plate(image_rgb, mask)

    def _carry(self, anny_vertices, match):
        """Anny vertices (fit pose) -> photo pixels, how much each faces the camera,
        camera-space points and normals. The first pairing with the SAM body is reused,
        so the same skin point is followed from the fitted body to a morphed one."""
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

    def _head(self, face_id):
        head = (face_id >= 0) & self.face_is_head[np.maximum(face_id, 0)]
        return cv2.GaussianBlur(head.astype(np.float32), (0, 0), max(2.0, 0.012 * max(self.size)))

    def _estimate_light(self, face_id, bary, normals):
        """Lighting of the photo as brightness = c0 + c . normal, fitted on bare skin.
        Falls back to soft light from above and in front when there is too little skin."""
        covered = face_id >= 0
        n = (normals[self.faces[np.maximum(face_id, 0)]] * bary[..., None]).sum(axis=2)
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

    def render(self, posed_vertices, start_relief: float, relief: float, passes: bool = False):
        """posed_vertices: the morphed Anny body in the fit pose (same frame as the
        vertices given at construction). Returns the re-drawn photo (RGB), or with
        passes=True a dict that also holds the depth, normal and albedo passes."""
        w, h = self.size
        xy, _, cam, normals, _ = self._carry(posed_vertices, self.match)
        face_id, bary = _rasterize(xy, cam[:, 2], self.faces, self.size)
        covered = face_id >= 0
        corners = self.faces[np.maximum(face_id, 0)]
        interp = lambda per_vertex: (per_vertex[corners] * bary[..., None]).sum(axis=2)
        source = interp(self.start_xy)                                        # where each pixel comes from
        gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        move = np.where(covered[..., None], source - np.stack([gx, gy], axis=-1), 0).astype(np.float32)
        # Carry the movement outwards to nearby pixels, fading with distance.
        dist, labels = cv2.distanceTransformWithLabels((~covered).astype(np.uint8), cv2.DIST_L2, 5,
                                                       labelType=cv2.DIST_LABEL_PIXEL)
        ys, xs = np.nonzero(covered)
        lookup = np.zeros(labels.max() + 1, dtype=np.int64)
        lookup[labels[ys, xs]] = np.arange(len(ys))
        near = (ys[lookup[labels]], xs[lookup[labels]])
        fade = np.exp(-dist / (0.04 * max(w, h)))[..., None]
        move = np.where(covered[..., None], move, move[near] * fade)
        # Identity lock: the head keeps its original pixels.
        lock = np.clip(np.maximum(self.start_head, self._head(face_id)) * 1.5, 0, 1)[..., None]
        move = cv2.GaussianBlur(move * (1 - lock), (0, 0), 1.5)
        mx, my = gx + move[..., 0], gy + move[..., 1]
        albedo = cv2.remap(self.image.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        alpha = cv2.remap(self.mask.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        alpha = cv2.GaussianBlur(alpha, (0, 0), 1.0)[..., None]
        skin = np.ones((h, w), np.float32) if self.skin is None else \
            cv2.remap(self.skin.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        skin = cv2.GaussianBlur(skin * covered * (1 - lock[..., 0]), (0, 0), 1.5)

        # Relief change, drawn in the same view from the UV maps.
        layout = self.app.layout
        n_tex = layout.size
        uv = (layout.uv[layout.uv_faces[np.maximum(face_id, 0)]] * bary[..., None]).sum(axis=2)
        tu = (uv[..., 0] * (n_tex - 1)).astype(np.float32)
        tv = ((1 - uv[..., 1]) * (n_tex - 1)).astype(np.float32)
        normal = interp(normals)
        normal /= np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-9
        person, change = albedo, relief - start_relief
        if change > 1e-3:
            # More defined than the photo: light the extra relief with the photo's own light.
            metres_per_px = float(np.median(cam[:, 2])) / self.pp.view["focal"]
            height = cv2.GaussianBlur(cv2.remap(self.app.height_m, tu, tv, cv2.INTER_LINEAR) * covered, (0, 0), 1.0)
            slope_y, slope_x = np.gradient(height * change / metres_per_px)
            bumped = normal - np.stack([slope_x, slope_y, np.zeros_like(slope_x)], axis=-1)
            bumped /= np.linalg.norm(bumped, axis=-1, keepdims=True) + 1e-9
            c0, c = self.light[0], self.light[1:]
            ratio = np.clip((c0 + bumped @ c).clip(0.05) / (c0 + normal @ c).clip(0.05), 0.72, 1.28)
            ratio = 1 + (cv2.GaussianBlur(ratio.astype(np.float32), (0, 0), 0.7) - 1) * skin
            person = albedo * ratio[..., None]
            normal = bumped
        elif change < -1e-3:
            # Less defined than the photo: let the photo's own definition fade.
            region = cv2.remap(self.app.relief_region, tu, tv, cv2.INTER_LINEAR) * skin
            k = cv2.GaussianBlur((min(1.0, -change * 1.6) * region).astype(np.float32), (0, 0), 2.0)[..., None]
            person = albedo * (1 - k) + cv2.GaussianBlur(albedo, (0, 0), 0.012 * max(w, h)) * k
        out = np.clip(self.plate.astype(np.float32) * (1 - alpha) + person * alpha, 0, 255).astype(np.uint8)
        if not passes:
            return out
        depth = np.where(covered, interp(cam[:, 2:3])[..., 0], np.inf).astype(np.float32)
        return {"image": out, "depth": depth, "covered": covered,
                "normal": np.where(covered[..., None], normal, 0).astype(np.float32),
                "albedo": np.clip(albedo, 0, 255).astype(np.uint8), "alpha": alpha[..., 0]}
