"""T9 copy of arena_compose.py (only change: url_of knows .usda assets). v21 room composer (Isaac Lab-Arena, Isaac Sim 6.2 environment): SimReady assets by pinned URL -> Lab-Arena relations
-> one flattened room USD that the Isaac Sim 6.1 harness opens (it adds walls, door, hatch, marks, robot, people, grades).
   <IsaacLab-Arena venv>/bin/python arena_compose.py --layout layouts/t1_or1.json --out rooms/t1_or1
Objects with a "pose" are anchors; objects with "on" are placed by the Lab-Arena solver on that anchor's top, no overlap.
Lab-Arena's own SimReady loader takes one rigid body only (10 of 11 T1 assets have joints), so every asset is wrapped here
the same way (pinned package download, Physics variant selected) and spawned as a plain USD object, joints kept."""
import argparse, hashlib, json, math, os, pathlib, time
ap = argparse.ArgumentParser()
ap.add_argument("--layout", required=True); ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--random-yaw", action="store_true")   # off: instruments stay aligned (random yaw put their ends on the T1 Mayo rim, 6 of 6 failed)
ap.add_argument("--variants", type=int, default=1)   # N > 1: N distinct layouts in one batched solve, random instrument yaw, one room per variant
A = ap.parse_args()
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
from pxr import Usd, UsdGeom, UsdPhysics
from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.assets.simready_central_asset_loader import _download_central_asset_package_to_cache
from isaaclab_arena.relations.object_placer import ObjectPlacer
from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.relations import IsAnchor, On, PositionLimitsBox
from isaaclab_arena.scene.scene import Scene
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.utils.usd.rigid_bodies import apply_usd_variant_selections

HERE = pathlib.Path(__file__).resolve().parent
L = json.load(open(HERE / A.layout)); OUT = HERE / A.out; OUT.mkdir(parents=True, exist_ok=True)
WRAP = OUT / "wrappers"; WRAP.mkdir(exist_ok=True)
T0 = time.time(); REC = {"layout": A.layout, "seed": A.seed, "commit": L["simready_commit"], "assets": {}, "placed": {}}

EXT = {"medical_sample_bag_a01": "usda"}      # T9: this asset ships as .usda at the pinned commit
def url_of(asset):
    return (f"https://huggingface.co/datasets/nvidia/simready-assets/resolve/{L['simready_commit']}/"
            f"{asset}/simready_usd/sm_{asset}_01.{EXT.get(asset, 'usd')}")

def wrap(asset):
    """Pinned package download + Physics variant selected (as the Lab-Arena loader does), without its one-body check."""
    url = url_of(asset); src = _download_central_asset_package_to_cache(url)
    st_src = Usd.Stage.Open(src)
    st = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(st, UsdGeom.GetStageMetersPerUnit(st_src)); UsdGeom.SetStageUpAxis(st, UsdGeom.GetStageUpAxis(st_src))
    root = st.DefinePrim("/Asset"); root.GetReferences().AddReference(src); st.SetDefaultPrim(root)
    apply_usd_variant_selections(st, {"Physics": "physics", "PhysX": "Enabled"})
    assert not st.GetCompositionErrors(), f"{asset}: composition errors"
    prims = list(Usd.PrimRange.Stage(st, Usd.TraverseInstanceProxies()))
    info = {"url": url, "rigid_bodies": sum(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in prims),
            "joints": sum(p.IsA(UsdPhysics.Joint) for p in prims),
            "articulation_root": any(p.HasAPI(UsdPhysics.ArticulationRootAPI) for p in prims)}
    rng = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(st.GetDefaultPrim()).ComputeAlignedRange()
    mpu = UsdGeom.GetStageMetersPerUnit(st); info["half_xy"] = [rng.GetSize()[0] * mpu / 2, rng.GetSize()[1] * mpu / 2]
    out = WRAP / f"{asset}.usda"; st.GetRootLayer().Export(str(out))
    info["wrapper"] = str(out); info["sha256"] = hashlib.sha256(out.read_bytes()).hexdigest()[:16]
    return str(out), info

def yaw_q(deg):
    h = math.radians(deg) / 2; return (0.0, 0.0, math.sin(h), math.cos(h))

objs = {}
for o in L["objects"]:
    path, info = wrap(o["asset"]); REC["assets"][o["name"]] = dict(info, asset=o["asset"])
    ob = Object(name=o["name"], usd_path=path, object_type=ObjectType.BASE)
    if "pose" in o:
        x, y, z, yaw = o["pose"]; ob.set_initial_pose(Pose(position_xyz=(x, y, z), rotation_xyzw=yaw_q(yaw))); ob.add_relation(IsAnchor())
    objs[o["name"]] = ob
for o in L["objects"]:
    if "on" in o:   # "edge_margin": keep an item this far inside the parent's top (a raised rim tips thin items off; T1 Mayo, 2026-10-28)
        objs[o["name"]].add_relation(On(objs[o["on"]], clearance_m=0.005, **({"edge_margin_m": o["edge_margin"]} if "edge_margin" in o else {})))
        fr = L.get("flat", {}).get(o["on"])   # measured flat top [x0, x1, y0, y1] (probes/flat_top.py): keep the whole item on it, 1 cm clear
        if fr:
            hx, hy = REC["assets"][o["name"]]["half_xy"]
            objs[o["name"]].add_relation(PositionLimitsBox(x_min=fr[0] + hx + 0.01, x_max=fr[1] - hx - 0.01, y_min=fr[2] + hy + 0.01, y_max=fr[3] - hy - 0.01))
print("EV wrapped", len(objs), "assets in", round(time.time() - T0, 1), "s", flush=True)

if A.variants > 1:   # seeds alone gave 2 distinct layouts in 5 (the solver keeps its best one); a batched solve gives one per env
    placer = ObjectPlacer(ObjectPlacerParams(placement_seed=A.seed, verbose=True, max_placement_attempts=50, random_yaw_init=A.random_yaw,
                                             apply_positions_to_objects=False, min_unique_layouts_per_env=A.variants))
    for i, r in enumerate(placer.place(list(objs.values()), num_envs=A.variants)):
        vout = pathlib.Path(str(OUT) + f"_v{i}"); vout.mkdir(parents=True, exist_ok=True); vrec = dict(REC, placed={}, variant=i, success=bool(r.success))
        for o in L["objects"]:
            ob = objs[o["name"]]
            if "on" in o:
                yaw = r.orientations.get(ob, 0.0); ob.set_initial_pose(Pose(position_xyz=tuple(float(v) for v in r.positions[ob]), rotation_xyzw=yaw_q(math.degrees(yaw))))
            p0 = ob.initial_pose; vrec["placed"][o["name"]] = {"xyz": [round(float(v), 4) for v in p0.position_xyz], "q_xyzw": [round(float(v), 5) for v in p0.rotation_xyzw]}
        Scene(assets=list(objs.values())).export_to_usd(vout / "room.usd"); json.dump(vrec, open(vout / "compose_record.json", "w"), indent=1)
        print("EV variant", i, "success", r.success, {k: v["xyz"][:2] for k, v in vrec["placed"].items() if k.startswith(("forceps", "scissors"))}, flush=True)
    app.close(); raise SystemExit(0)
placer = ObjectPlacer(ObjectPlacerParams(placement_seed=A.seed, verbose=True, max_placement_attempts=50))
res = placer.place(list(objs.values()))[0]
REC["placement_success"] = bool(res.success)
for o in L["objects"]:
    p = objs[o["name"]].initial_pose
    pos = [round(float(v), 4) for v in p.position_xyz]; q = [round(float(v), 5) for v in p.rotation_xyzw]
    REC["placed"][o["name"]] = {"xyz": pos, "q_xyzw": q, "how": "anchor" if "pose" in o else f"on {o['on']}"}
    print("EV placed", o["name"], pos, q, flush=True)

Scene(assets=list(objs.values())).export_to_usd(OUT / "room.usd")
REC["room_usd"] = str(OUT / "room.usd"); REC["seconds"] = round(time.time() - T0, 1)
json.dump(REC, open(OUT / "compose_record.json", "w"), indent=1)
print("EV room", OUT / "room.usd", "success", res.success, "s", REC["seconds"], flush=True)
app.close()
