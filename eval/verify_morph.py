"""Quick check of stage 4: build a body with a 950 mm waist, then morph it to 820 mm.

Run:  py eval/verify_morph.py
Writes outputs/verify_morph/waist_950.glb and waist_820.glb.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stage2_measure import GIRTH_KEYS, AnnyBody  # noqa: E402
from stage4_morph import export, solve  # noqa: E402

SEX, AGE, HEIGHT_CM = "male", 35, 178.0
OUT = ROOT / "outputs" / "verify_morph"

t0 = time.time()
anny_body = AnnyBody()
print(f"Anny loaded in {time.time() - t0:.0f}s\n")

print("Solve 1: start from an average body, target ISO waist = 950 mm")
d1, l1, body1, g1, _ = solve(anny_body, SEX, AGE, HEIGHT_CM, {"waist": 950.0}, log=print)
export(anny_body, body1, d1, l1, OUT, "waist_950")

print("\nSolve 2: start from that body, target ISO waist = 820 mm")
t1 = time.time()
d2, l2, body2, g2, _ = solve(anny_body, SEX, AGE, HEIGHT_CM, {"waist": 820.0},
                             init_dials=d1, init_local=l1, log=print)
print(f"(solve took {time.time() - t1:.0f}s)")
export(anny_body, body2, d2, l2, OUT, "waist_820")


def height_cm(body):
    z = body["vertices"][:, 2]
    return float(z.max() - z.min()) * 100


print(f"\n{'':14s}{'before':>10s}{'after':>10s}{'change':>10s}")
print(f"{'height cm':14s}{height_cm(body1):10.2f}{height_cm(body2):10.2f}{height_cm(body2) - height_cm(body1):10.2f}")
m1, m2 = float(body1["mass_kg"]), float(body2["mass_kg"])
print(f"{'weight kg':14s}{m1:10.1f}{m2:10.1f}{m2 - m1:10.1f}")
for k in GIRTH_KEYS:
    print(f"{k + ' mm':14s}{g1[k]:10.0f}{g2[k]:10.0f}{g2[k] - g1[k]:10.0f}")
print("\ndials before", {k: round(v, 3) for k, v in d1.items()})
print("dials after ", {k: round(v, 3) for k, v in d2.items()})
print("local after ", {k: round(v, 3) for k, v in l2.items()})
print(f"\nMeshes written to {OUT}")
