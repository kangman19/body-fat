"""Local test GUI. Run:  py src/app.py   then open http://127.0.0.1:8000

Everything stays on this machine: the server only listens on 127.0.0.1 and the page
loads three.js from src/static/vendor, not from the internet.

Endpoints
    GET  /api/faces     triangle list (sent once)
    POST /api/body      height, weight, sex, age -> generic Anny body + body fat estimate
    POST /api/photo     photo(s) + height, weight, sex, age -> Anny body fitted to the photo,
                        textured, plus the photo re-drawn for that body
    POST /api/bodyfat   that body + a chosen body fat % -> morphed body (stages 3 + 4)
"""
from __future__ import annotations

import base64
import gc
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage2_measure import DIAL_RANGES, AnnyBody  # noqa: E402
import stage1_mesh as stage1  # noqa: E402
import stage3_nhanes as stage3  # noqa: E402
import stage5_photoreal as stage5  # noqa: E402
from stage4_morph import fixed_for_bf, muscle_dial_for_bf, relief, solve  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"

print("Loading Anny (a few seconds, or several minutes the very first time)...")
anny_body = AnnyBody()
bf_model = stage3.load_model()
appearance = stage5.Appearance(anny_body)
# The latest photo build (one user, one machine): what is needed to re-draw the photo
# and its texture when the slider moves. Lives in memory only.
photo_session: dict | None = None
lock = threading.Lock()  # one body computation at a time
app = FastAPI()


class Person(BaseModel):
    sex: str = Field(pattern="^(male|female)$")
    age: float = Field(ge=18, le=90)
    height_cm: float = Field(ge=130, le=220)


class BodyRequest(Person):
    weight_kg: float = Field(ge=35, le=200)


class BodyFatRequest(BodyRequest):
    """weight_kg and dials describe the generated (starting) body, not the last morph."""
    dials: dict[str, float]
    local: dict[str, float] = {}
    from_photo: bool = False
    target_bf: float = Field(ge=3, le=60)


def _b64(array, dtype):
    return base64.b64encode(np.ascontiguousarray(array, dtype=dtype).tobytes()).decode()


def _jpeg(image_rgb, quality=90):
    ok, data = cv2.imencode(".jpg", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(data.tobytes()).decode()


def _payload(person: Person, dials, local, body, started, bf=None, texture=None):
    """Everything the page needs to draw and describe one body."""
    with torch.no_grad():
        girths = {k: float(x) for k, x in anny_body.measure(body).items()}
    v = body["vertices"].detach().numpy()
    mass = float(body["mass_kg"])
    half = bf_model[person.sex]["rmse"]
    if bf is None:  # no chosen value: estimate it from the measurements
        bf = stage3.estimate_bf(bf_model, person.sex, person.age, person.height_cm, mass, girths)["bf_mid"]
    return {
        "vertices": _b64(np.stack([v[:, 0], v[:, 2], -v[:, 1]], axis=1), np.float32),  # Y-up
        "dials": {k: float(x) for k, x in dials.items()},
        "local": {k: float(x) for k, x in local.items()},
        "weight_kg": round(mass, 1),
        "bmi": round(mass / (person.height_cm / 100) ** 2, 1),
        "girths_mm": {k: round(x) for k, x in girths.items()},
        "bf_mid": round(bf, 1),
        "bf_range": [round(bf - half, 1), round(bf + half, 1)],
        "bf_floor": bf_model[person.sex]["bf_floor"],
        "bf_ceiling": stage3.BF_CEILING[person.sex],
        # Skin image in Anny's UV layout: plain clay unless a photo texture is given.
        # "relief" (0..1) is the strength of muscle relief for this sex and body fat;
        # the viewer uses it for vertex displacement and for the normal map.
        "relief": round(relief(person.sex, bf), 3),
        "texture": _jpeg(texture if texture is not None else appearance.clay()),
        "seconds": round(time.time() - started, 1),
    }


def _clamp(dials):
    return {k: min(max(float(dials[k]), lo), hi) for k, (lo, hi) in DIAL_RANGES.items()}


@app.get("/api/faces")
def faces():
    layout = appearance.layout
    png = lambda data: "data:image/png;base64," + base64.b64encode(data).decode()
    return {
        # The real mesh (for smooth normals) and how the viewer's finer mesh derives from
        # it: one vertex per UV point, then split twice; each new vertex is the average
        # of its two parents.
        "faces": _b64(anny_body.faces.numpy(), np.uint32),
        "uv_to_vertex": _b64(layout.uv_to_vertex, np.uint32),
        "parents1": _b64(layout.parents1, np.uint32),
        "parents2": _b64(layout.parents2, np.uint32),
        "fine_faces": _b64(layout.fine_faces, np.uint32),
        "fine_uv": _b64(layout.fine_uv, np.float32),
        # Muscle relief: the same height field as displacement and as a normal map.
        "displacement": png(appearance.displacement_png()),
        "normal": png(appearance.normal_png()),
        "relief_range_m": stage5.RELIEF_RANGE_M,
        "normal_scale_full": stage5.NORMAL_SCALE_FULL,
    }


@app.post("/api/body")
def make_body(req: BodyRequest):
    global photo_session
    started = time.time()
    with lock:
        photo_session = None
        dials, body = anny_body.from_height_weight_sex(
            req.height_cm, req.weight_kg, req.sex, req.age)
        return _payload(req, dials, {}, body, started)


@app.post("/api/photo")
def photo_body(
    front: UploadFile = File(...),
    side: UploadFile | None = File(None),
    sex: str = Form(...), age: float = Form(...),
    height_cm: float = Form(...), weight_kg: float = Form(...),
):
    """Photos are read into memory and never written to disk. The latest photo is kept
    in memory (photo_session) so it can be re-drawn when the slider moves."""
    global photo_session
    started = time.time()
    try:
        req = BodyRequest(sex=sex, age=age, height_cm=height_cm, weight_kg=weight_kg)
        images = [stage1.decode_image(f.file.read()) for f in (front, side) if f is not None]
        with lock:
            # SAM 3D Body is loaded for this photo only and freed as soon as it has read
            # it, so the big models are never all in memory together.
            print("Loading SAM 3D Body (about 40 seconds)...")
            fit = stage1.fit_from_photos(stage1.SamBody(), anny_body, bf_model, images, req.height_cm,
                                         req.weight_kg, req.sex, req.age, log=print)
            gc.collect()
            start_relief = relief(req.sex, fit["bf_mid"])
            print("Painting the photo onto the body and removing the person from the background...")
            view = stage5.PhotoView(images[0], fit["mask"], fit["skin"], fit["photo_pose"], anny_body,
                                    fit["posed_vertices"], appearance)
            texture, skin_texels, _ = appearance.bake_photo(
                images[0], fit["mask"], fit["skin"], view.start_xy, view.facing, fit["face_colour"])
            photo_session = {"view": view, "texture": texture, "skin_texels": skin_texels,
                             "pose": fit["pose"], "trans": fit["trans"], "relief": start_relief}
            reply = _payload(req, fit["dials"], fit["local"], fit["body"], started, texture=texture)
            reply["photo_view"] = _jpeg(view.render(fit["posed_vertices"], start_relief, start_relief), 93)
            reply["fit_report"] = fit["report"]
            reply["surface_error_mm"] = round(fit["surface_error_mm"], 1)
            reply["photo_girths_mm"] = {k: round(v) for k, v in fit["photo_girths_mm"].items()}
            reply["photos_used"] = len(images)
    except ValueError as err:
        raise HTTPException(status_code=422, detail=str(err))
    return reply


@app.post("/api/bodyfat")
def body_fat(req: BodyFatRequest):
    started = time.time()
    with lock:
        dials = _clamp(req.dials)
        with torch.no_grad():
            start = anny_body.build(req.sex, req.age, req.height_cm, dials, req.local)
            girths = {k: float(x) for k, x in anny_body.measure(start).items()}
            start_mass = float(start["mass_kg"])
        start_bf = stage3.estimate_bf(bf_model, req.sex, req.age, req.height_cm, start_mass, girths)["bf_mid"]
        t = stage3.targets_for_bf(bf_model, req.sex, req.age, req.height_cm,
                                  start_mass, girths, req.target_bf)
        targets = {**t["girths_mm"], "mass_kg": t["mass_kg"]}
        # Body fat also drives muscle: abdominal tautness is set for the target, the
        # person's regional muscle sliders and muscle dial are nudged the same way.
        muscle = muscle_dial_for_bf(req.sex, dials["muscle"], start_bf, t["target_bf"])
        new_dials, local, body, _, _ = solve(
            anny_body, req.sex, req.age, req.height_cm, targets,
            init_dials=dials, init_local=req.local, anchor_local=True,
            fixed_local=fixed_for_bf(req.sex, t["target_bf"], req.local, start_bf),
            muscle_prior=(muscle, 0.05),
            steps=300)
        texture = photo_view = None
        session = photo_session if req.from_photo else None
        if session is not None:   # re-draw the photo and its texture for the new body
            target_relief = relief(req.sex, t["target_bf"])
            with torch.no_grad():
                posed = anny_body.build(req.sex, req.age, req.height_cm, new_dials, local, session["pose"])
            posed = (posed["posed_vertices"] + session["trans"]).numpy()
            photo_view = session["view"].render(posed, session["relief"], target_relief)
            texture = appearance.restyle(session["texture"], session["skin_texels"],
                                         session["relief"], target_relief)
        reply = _payload(req, new_dials, local, body, started, bf=t["target_bf"], texture=texture)
        if photo_view is not None:
            reply["photo_view"] = _jpeg(photo_view, 93)
        reply["targets"] = {k: round(v, 1) for k, v in targets.items()}
        return reply


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
