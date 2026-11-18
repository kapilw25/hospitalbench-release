"""v21 T2 room composer (Isaac Lab-Arena, Isaac Sim 6.2 environment): copy of arena_compose.py reduced to anchors, plus a per-object "scale" (uniform).
   <IsaacLab-Arena venv>/bin/python t2_arena_compose.py --layout layouts/t2_dialysis.json --out rooms/t2_dialysis
Every object has a "pose" [x, y, z, yaw deg] and may have "scale". No Lab-Arena solver (all poses are explicit); the wrapper step is the same as arena_compose.py."""
import argparse, hashlib, json, math, pathlib, time
ap = argparse.ArgumentParser(); ap.add_argument("--layout", required=True); ap.add_argument("--out", required=True)
A = ap.parse_args()
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
from pxr import Usd, UsdGeom, UsdPhysics
from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.assets.simready_central_asset_loader import _download_central_asset_package_to_cache
from isaaclab_arena.relations.relations import IsAnchor
from isaaclab_arena.scene.scene import Scene
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.utils.usd.rigid_bodies import apply_usd_variant_selections
HERE = pathlib.Path(__file__).resolve().parent
L = json.load(open(HERE / A.layout)); OUT = HERE / A.out; OUT.mkdir(parents=True, exist_ok=True)
WRAP = OUT / "wrappers"; WRAP.mkdir(exist_ok=True)
T0 = time.time(); REC = {"layout": A.layout, "commit": L["simready_commit"], "assets": {}, "placed": {}}
def url_of(asset): return f"https://huggingface.co/datasets/nvidia/simready-assets/resolve/{L['simready_commit']}/{asset}/simready_usd/sm_{asset}_01.usd"
def wrap(asset):
    url = url_of(asset); src = _download_central_asset_package_to_cache(url)
    st_src = Usd.Stage.Open(src); st = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(st, UsdGeom.GetStageMetersPerUnit(st_src)); UsdGeom.SetStageUpAxis(st, UsdGeom.GetStageUpAxis(st_src))
    root = st.DefinePrim("/Asset"); root.GetReferences().AddReference(src); st.SetDefaultPrim(root)
    apply_usd_variant_selections(st, {"Physics": "physics", "PhysX": "Enabled"})
    assert not st.GetCompositionErrors(), f"{asset}: composition errors"
    prims = list(Usd.PrimRange.Stage(st, Usd.TraverseInstanceProxies()))
    info = {"url": url, "rigid_bodies": sum(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in prims), "joints": sum(p.IsA(UsdPhysics.Joint) for p in prims)}
    out = WRAP / f"{asset}.usda"; st.GetRootLayer().Export(str(out)); info["wrapper"] = str(out); info["sha256"] = hashlib.sha256(out.read_bytes()).hexdigest()[:16]
    return str(out), info
def yaw_q(deg): h = math.radians(deg) / 2; return (0.0, 0.0, math.sin(h), math.cos(h))
for a_ in L.get("wrap_only", []): wrap(a_)      # loose props placed by the harness (t2_props.spawn): wrapper files only, not in the room
objs = []
for o in L["objects"]:
    path, info = wrap(o["asset"]); REC["assets"][o["name"]] = dict(info, asset=o["asset"])
    s = float(o.get("scale", 1.0))
    ob = Object(name=o["name"], usd_path=path, object_type=ObjectType.BASE, scale=(s, s, s))
    x, y, z, yaw = o["pose"]; ob.set_initial_pose(Pose(position_xyz=(x, y, z), rotation_xyzw=yaw_q(yaw))); ob.add_relation(IsAnchor())
    objs.append(ob); REC["placed"][o["name"]] = {"xyz": [x, y, z], "yaw": yaw, "scale": s}
print("EV wrapped", len(objs), "assets in", round(time.time() - T0, 1), "s", flush=True)
Scene(assets=objs).export_to_usd(OUT / "room.usd")
REC["seconds"] = round(time.time() - T0, 1); json.dump(REC, open(OUT / "compose_record.json", "w"), indent=1)
print("EV room", OUT / "room.usd", "s", REC["seconds"], flush=True)
app.close()
