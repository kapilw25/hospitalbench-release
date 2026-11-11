"""T7 asset fixes (Isaac Sim 6.1 harness side), applied to the T7 warehouse room USD after it is referenced and before physics starts.
Own file (the shared v21/asset_fixes.py is T1's).  The SimReady pallet jack ships with free joints and no masses; every fix below stands in for a real property:
  masses            jack steel parts, pallets 14 kg, glove boxes 0.25 kg (PhysX would otherwise take density x mesh volume)
  wheels            HOLONOMIC JACK (design change, t7_design_changes.md): the shipped wheel colliders are switched off and each wheel body gets small skid spheres (r 1 cm) at its floor contact points, on a
                    low-friction material (mu 0.03, combine mode min = a hospital caster's rolling resistance); wheel spin, rear swivel, handle tilt and the shipped lever are locked.  The jack then moves in
                    any direction under the hand that holds its handle, like the T1 trolleys on free casters.
  lift              the three front roller joints and the swivel joint become D6 joints with a free vertical slide and a position drive: the whole body (forks, housing) rises by
                    `lift` m relative to the wheels, as on a real jack (the forks lift, the wheels stay on the floor).  The drive target is set per physics step by t7_world.
  colliders         jack and pallet meshes: convexDecomposition (the shipped sdf meshes made resting contact noisy on T1's trolleys)
  shelves           steel shelf bays: box colliders (4 boards + 4 posts) instead of the shipped SDF mesh collider (a glove box resting on the board tumbled off by itself)
Everything here is a property of the prop, not of the robot: no robot gain, force cap or mass is touched."""
import math
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Gf, Sdf

MASS = {"sm_mainbody": 28.0, "sm_rotatingpart": 3.0, "sm_handlemain": 2.0, "sm_handle": 0.15, "sm_frontwheel1": 1.5, "sm_frontwheel2": 1.5,
        "sm_frontwheel3": 0.4, "sm_backwheel": 1.0}
PALLET_KG, BOX_KG = 14.0, 0.25
SKID_MU = 0.005; SKID_R = 0.01
HANDLE_K, HANDLE_D = 3.0, 0.5            # Nm / deg, Nm s / deg   (handle tilt hold)
LEVER_K, LEVER_D = 0.05, 0.01
LIFT_K, LIFT_D, LIFT_F = 1.0e6, 1.0e4, 1.0e6   # D6 vertical drive per joint (1e5 / 4000 N could not raise the rear: measured 2026-10-30, only the front lifted)
LIFT_MAX = 0.10
SWIVEL_LIM_DEG = 60.0
JACK_DAMP = 0.3                       # linear and angular damping of the jack bodies (1/s)
FLEX = None


def _drive(p, kind, k, d, target=0.0, maxf=1e6):
    dr = UsdPhysics.DriveAPI.Apply(p, kind); dr.CreateTypeAttr("force"); dr.CreateStiffnessAttr(k); dr.CreateDampingAttr(d); dr.CreateMaxForceAttr(maxf); dr.CreateTargetPositionAttr(target)


def _coulomb(p, tc, w0_deg):
    dr = UsdPhysics.DriveAPI.Apply(p, "angular"); dr.CreateTypeAttr("force"); dr.CreateStiffnessAttr(0.0); dr.CreateDampingAttr(tc / w0_deg); dr.CreateMaxForceAttr(tc)


def _z_ext(prim):
    zs = []
    for c in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if c.IsA(UsdGeom.Mesh):
            pts = UsdGeom.Mesh(c).GetPointsAttr().Get(); M = UsdGeom.Xformable(c).ComputeLocalToWorldTransform(0)
            zs += [M.Transform(v)[2] for v in (pts or [])]
    return (max(zs) - min(zs)) if zs else 0.1


def _set_mass(b, kg):
    UsdPhysics.MassAPI.Apply(b).CreateMassAttr(kg)


def _convex(root):
    n = 0
    for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if p.HasAPI(UsdPhysics.CollisionAPI) and p.IsA(UsdGeom.Mesh):
            UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr("convexDecomposition"); n += 1
    return n


def _lock(P, ax):
    l = UsdPhysics.LimitAPI.Apply(P, ax); l.CreateLowAttr(1.0); l.CreateHighAttr(-1.0)           # low > high = locked


def _d6_slide(st, jprim):
    """replace a revolute joint by a D6 joint (same bodies and frames) with every rotation locked and a free vertical slide (transZ, limit [-LIFT_MAX, 0], position drive)"""
    j = UsdPhysics.Joint(jprim); b0 = j.GetBody0Rel().GetTargets(); b1 = j.GetBody1Rel().GetTargets()
    lp0, lp1 = j.GetLocalPos0Attr().Get(), j.GetLocalPos1Attr().Get(); lr0, lr1 = j.GetLocalRot0Attr().Get(), j.GetLocalRot1Attr().Get()
    path = str(jprim.GetPath()).replace("/joint_", "/lift_joint_"); jprim.SetActive(False)
    d6 = UsdPhysics.Joint.Define(st, path)
    d6.CreateBody0Rel().SetTargets(b0); d6.CreateBody1Rel().SetTargets(b1)
    d6.CreateLocalPos0Attr(lp0); d6.CreateLocalPos1Attr(lp1); d6.CreateLocalRot0Attr(lr0); d6.CreateLocalRot1Attr(lr1)
    P = d6.GetPrim()
    for ax in ("transX", "transY", "rotX", "rotY", "rotZ"): _lock(P, ax)
    lz = UsdPhysics.LimitAPI.Apply(P, "transZ"); lz.CreateLowAttr(-LIFT_MAX); lz.CreateHighAttr(0.0)
    _drive(P, "transZ", LIFT_K, LIFT_D, 0.0, LIFT_F)
    return path


# skid contact points, jack-local (x, y, z of the sphere centre), per wheel body stem
SKIDS = {"sm_frontwheel1": [(-0.20, -1.17), (0.20, -1.17)], "sm_frontwheel2": [(-0.20, -1.27), (0.20, -1.27)], "sm_backwheel": [(0.0, 0.0)]}


def _skids(st, top, body, pts):
    from pxr import PhysxSchema
    mat = UsdPhysics.MaterialAPI.Apply(UsdShade_define(st, "/World/Room/_t7_skid_pm").GetPrim())
    mat.CreateStaticFrictionAttr(SKID_MU); mat.CreateDynamicFrictionAttr(SKID_MU); mat.CreateRestitutionAttr(0.0)
    PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim()).CreateFrictionCombineModeAttr("min")
    Mj = UsdGeom.Xformable(top).ComputeLocalToWorldTransform(0); Mb = UsdGeom.Xformable(body).ComputeLocalToWorldTransform(0)
    for p in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
        if p.HasAPI(UsdPhysics.CollisionAPI): UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Set(False)
    for k, (x, y) in enumerate(pts):
        w = Mj.Transform(Gf.Vec3d(x, y, SKID_R)); loc = Mb.GetInverse().Transform(w)
        sp = UsdGeom.Sphere.Define(st, f"{body.GetPath()}/skid{k}"); sp.GetRadiusAttr().Set(SKID_R)
        UsdGeom.Imageable(sp).MakeInvisible(); xf = UsdGeom.Xformable(sp); xf.ClearXformOpOrder(); xf.AddTranslateOp().Set(Gf.Vec3d(*loc))
        UsdPhysics.CollisionAPI.Apply(sp.GetPrim()); UsdShade.MaterialBindingAPI.Apply(sp.GetPrim()).Bind(UsdShade.Material(st.GetPrimAtPath("/World/Room/_t7_skid_pm")), UsdShade.Tokens.weakerThanDescendants, "physics")


def UsdShade_define(st, path):
    return UsdShade.Material.Define(st, path)


PALLET_TRIM = 0.06             # deck and blocks stop 6 cm short of both ends along the entry axis: the jack housing front then never meets the pallet edge (fully in, tip gap about -7 cm made the housing push the pallet)
PALLET_BLOCK_H = 0.115          # underside of the deck above the floor (the tines are 0.095 high: 2 cm of clearance)


def _pallet_boxes(st, top, body):
    """Pallet collision = plain boxes (design change 2026-10-30, t7_design_changes.md): the shipped block pallet has 3 cm bottom boards across the fork channels, which stopped the jack's wheels and shoved the pallet.
    The mesh colliders are switched off; a top deck box and six 10 x 10 cm blocks (3 along the entry axis x 2 side rows; one 80 cm wide channel between the rows) stand in their place, deck underside at PALLET_BLOCK_H."""
    ps = []
    for p in Usd.PrimRange(top, Usd.TraverseInstanceProxies()):
        if p.IsA(UsdGeom.Mesh):
            M = UsdGeom.Xformable(p).ComputeLocalToWorldTransform(0)
            for v in (UsdGeom.Mesh(p).GetPointsAttr().Get() or []): ps.append(list(M.Transform(Gf.Vec3d(float(v[0]), float(v[1]), float(v[2])))))
            if p.HasAPI(UsdPhysics.CollisionAPI): UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Set(False)
    import numpy as np
    # extents in the BODY frame (state 1 has the pallets turned 94 and 87 degrees: the world box of a turned pallet is 7-8 cm larger each way, and boxes built from it were 7 cm longer than the pallet,
    # which undid the end trim: the housing met the pallet at tip gap -7.6 cm and shoved it; REPL 2026-11-01).  Lengths below are metres along the body axes (scale undone per axis).
    Mb = UsdGeom.Xformable(body).ComputeLocalToWorldTransform(0); Mi = Mb.GetInverse(); Mn = np.array(Mb).reshape(4, 4)
    sc = np.array([np.linalg.norm(Mn[i, :3]) for i in range(3)])      # row-vector convention: row i is body axis i
    Q = np.array([[q[0], q[1], q[2]] for q in (Mi.Transform(Gf.Vec3d(*[float(v) for v in pt])) for pt in ps)]) * sc      # body-frame metres
    lo, hi = Q.min(0), Q.max(0); c = (lo + hi) / 2
    la = 0 if (hi[0] - lo[0]) >= (hi[1] - lo[1]) else 1      # the long (entry) axis
    sa = 1 - la
    def box(name, blo, bhi):
        l, h = np.array(blo) / sc, np.array(bhi) / sc
        cu = UsdGeom.Cube.Define(st, f"{body.GetPath()}/{name}"); cu.GetSizeAttr().Set(1.0); UsdGeom.Imageable(cu).MakeInvisible()
        xf = UsdGeom.Xformable(cu); xf.ClearXformOpOrder(); xf.AddTranslateOp().Set(Gf.Vec3d(*((l + h) / 2))); xf.AddScaleOp().Set(Gf.Vec3f(*(h - l)))
        UsdPhysics.CollisionAPI.Apply(cu.GetPrim())
    def span(a_lo, a_hi, s_lo, s_hi, z_lo, z_hi):
        L, H = [0.0] * 3, [0.0] * 3; L[la], H[la] = a_lo, a_hi; L[sa], H[sa] = s_lo, s_hi; L[2], H[2] = z_lo, z_hi; return L, H
    box("deck", *span(lo[la] + PALLET_TRIM, hi[la] - PALLET_TRIM, lo[sa], hi[sa], PALLET_BLOCK_H, hi[2]))
    for i, bx in enumerate((lo[la] + PALLET_TRIM + 0.05, c[la], hi[la] - PALLET_TRIM - 0.05)):
        for j, by in enumerate((lo[sa] + 0.05, hi[sa] - 0.05)):      # two side rows only: the jack's tine collider is one hull over both tines and would hit a middle row
            box(f"blk{i}{j}", *span(bx - 0.05, bx + 0.05, by - 0.05, by + 0.05, 0.0, PALLET_BLOCK_H))
    return 7


SHELF_TOP0, SHELF_PITCH, SHELF_LEVELS, SHELF_SLAB, SHELF_POST = 0.6422, 0.6, 3, 0.02, 0.07


def _shelf_boxes(st, top):
    """Shelf bay collision = plain boxes (design change 2026-11-01, t7_design_changes.md): the shipped steel shelf collides as an SDF mesh, and a glove box resting on its board rocked, rose 7 cm and tumbled to the
    floor with no hand near it, at some spots (PhysX SDF resolution; T7.04 failed 4 of 5 times).  The mesh collider is switched off; one 2 cm slab per board (tops 0.642, 1.242, 1.842 m above the shelf foot, the top
    plate at 2.40 m) and four 7 cm corner posts stand in its place.  The shelf looks the same."""
    ps = []
    for p in Usd.PrimRange(top, Usd.TraverseInstanceProxies()):
        if p.IsA(UsdGeom.Mesh):
            M = UsdGeom.Xformable(p).ComputeLocalToWorldTransform(0)
            for v in (UsdGeom.Mesh(p).GetPointsAttr().Get() or []): ps.append(list(M.Transform(Gf.Vec3d(float(v[0]), float(v[1]), float(v[2])))))
            if p.HasAPI(UsdPhysics.CollisionAPI): UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Set(False)
    import numpy as np
    P = np.array(ps); lo, hi = P.min(0), P.max(0)
    Mi = UsdGeom.Xformable(top).ComputeLocalToWorldTransform(0).GetInverse()
    def box(name, wlo, whi):
        corners = [Mi.Transform(Gf.Vec3d(x, y, z)) for x in (wlo[0], whi[0]) for y in (wlo[1], whi[1]) for z in (wlo[2], whi[2])]
        arr = np.array([[q[0], q[1], q[2]] for q in corners]); l, h = arr.min(0), arr.max(0)
        cu = UsdGeom.Cube.Define(st, f"{top.GetPath()}/{name}"); cu.GetSizeAttr().Set(1.0); UsdGeom.Imageable(cu).MakeInvisible()
        xf = UsdGeom.Xformable(cu); xf.ClearXformOpOrder(); xf.AddTranslateOp().Set(Gf.Vec3d(*((l + h) / 2))); xf.AddScaleOp().Set(Gf.Vec3f(*(h - l)))
        UsdPhysics.CollisionAPI.Apply(cu.GetPrim())
    tops = [lo[2] + SHELF_TOP0 + SHELF_PITCH * k for k in range(SHELF_LEVELS)] + [hi[2] - 0.038]
    for k, zt in enumerate(tops): box(f"slab{k}", (lo[0], lo[1], zt - SHELF_SLAB), (hi[0], hi[1], zt))
    for i, cx in enumerate((lo[0], hi[0] - SHELF_POST)):
        for j, cy in enumerate((lo[1], hi[1] - SHELF_POST)): box(f"post{i}{j}", (cx, cy, lo[2]), (cx + SHELF_POST, cy + SHELF_POST, hi[2]))
    return len(tops) + 4


def apply(st, layout, root="/World/Room", lift_d6=True):
    out = {}
    fixed = {o["name"] for o in layout["objects"] if o.get("fixed")}
    for o in layout["objects"]:
        top = st.GetPrimAtPath(f"{root}/{o['name']}"); a = o["asset"]
        for p in Usd.PrimRange(top):
            if p.IsInstance(): p.SetInstanceable(False)
        if a.startswith("shelve"):
            out[o["name"]] = dict(collider=_shelf_boxes(st, top)); continue
        if o["name"] in fixed: continue
        n = {}
        bodies = {p.GetName(): p for p in Usd.PrimRange(top) if p.HasAPI(UsdPhysics.RigidBodyAPI)}
        if a.startswith("pallet_block"):
            for b in bodies.values(): _set_mass(b, PALLET_KG)
            n["collider"] = _pallet_boxes(st, top, next(iter(bodies.values()))); n["mass"] = PALLET_KG
        elif a.startswith("box_glove"):
            for b in bodies.values(): _set_mass(b, BOX_KG)
            n["mass"] = BOX_KG
        elif a.startswith("palletJack"):
            tot = 0.0
            for nm, b in bodies.items():
                k = next((v for key, v in MASS.items() if nm.startswith(key + "_")), 1.0); _set_mass(b, k); tot += k
            n["mass"] = tot; n["collider"] = _convex(top)
            made = []; sk = 0
            for nm, b in bodies.items():
                for stem, pts in SKIDS.items():
                    if nm.startswith(stem + "_"): _skids(st, top, b, pts); sk += len(pts)
                if nm.startswith("sm_frontwheel3_"):      # nose roller: 2.3 cm above the floor, no contact needed
                    for p in Usd.PrimRange(b, Usd.TraverseInstanceProxies()):
                        if p.HasAPI(UsdPhysics.CollisionAPI): UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Set(False)
            for p in list(Usd.PrimRange(top)):
                if not p.IsA(UsdPhysics.RevoluteJoint): continue
                name = p.GetName()
                if "frontwheel" in name or "rotatingpart" in name:
                    if lift_d6: made.append(_d6_slide(st, p))
                    else: p.GetAttribute("physics:lowerLimit").Set(0.0); p.GetAttribute("physics:upperLimit").Set(0.0)
                else: p.GetAttribute("physics:lowerLimit").Set(0.0); p.GetAttribute("physics:upperLimit").Set(0.0)      # back wheel spin, handle tilt, shipped lever: locked
            from pxr import PhysxSchema
            for nm, b in bodies.items():      # rolling resistance: a touch must not send the free-rolling jack coasting (measured 2026-10-30: an arm sweep moved it 4 cm and 2 deg)
                api = PhysxSchema.PhysxRigidBodyAPI.Apply(b); api.CreateLinearDampingAttr(JACK_DAMP); api.CreateAngularDampingAttr(JACK_DAMP)
            n["skids"] = sk; n["lift_d6"] = len(made)
        out[o["name"]] = n
    return out
