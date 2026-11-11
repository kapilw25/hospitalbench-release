"""v21 asset fixes (Isaac Sim 6.1 harness side), applied to a Lab-Arena room USD after it is referenced and before physics
starts. SimReady jointed assets ship with free joints (no drive, no friction): a free drawer slides and its reaction rolls
the trolley on free casters (Mayo b03 drifted 3-5 cm at 120 Hz in an empty room), and hinged instruments with the shipped
sdf colliders were thrown off the Mayo (2026-10-28 room checks). Each fix below stands in for a real property:
  drawer and slide joints   -> held at their authored position by a soft linear drive (a drawer catch)
  caster swivel and roll    -> velocity damping only (rolling resistance; the cart still rolls when pushed)
  instrument hinges         -> held at the authored (closed) angle 0 by a soft angular drive (forceps are clamped shut in use;
                               the lower limit is NOT closed: forceps range -130..8 deg, scissors 0..130 deg)
  other small hinges        -> held at their authored angle (tray clips and locks, a trolley's lever handle)
  instrument colliders      -> convexDecomposition instead of sdf (thin 5 mm parts)
Fixed (kinematic) objects get no drives: their joints are disabled by the caller."""
import math, os
from pxr import Usd, UsdGeom, UsdPhysics

LIN_K, LIN_D = 500.0, 50.0            # N/m, N s/m
HOLD_K, HOLD_D = 2.0, 0.2             # per degree
# T1_TOOL_K_SCALE (default 1.0 = unchanged): scale of BOTH instrument-hinge drive gains (spring and damper).  Measured 2026-10-30 (forceps start-fix helper, chain helper): the stiff force drives on 0.1-0.14 kg
# bodies jitter an awake forceps at 5-34 rad/s and a scissors dropped 3 mm beside it throws it 2-209 cm; x0.1 on both tools keeps it still.
_TKS = float(os.environ.get("T1_TOOL_K_SCALE", "1.0"))
TOOL_K, TOOL_D = 5.0 * _TKS, 0.5 * _TKS             # per degree
ROLL_D = 0.02                         # per degree/s, no spring
# T1_LEVER_LATCH=1 (default 0, maintainer option 1 for sub-task 8, 2026-10-28): the transport trolley's lever (handle_01) becomes a two-detent latch (LOCKED 0 deg, UNLOCKED LATCH_U_DEG) with a
# real pull range and a soft detent drive; the detents and the slide hold are driven each physics step by t1_world.install_lever_latch.  The knob no longer collides with the chassis, the
# slide column or the lock part it sits against (collision filter), uses a convex hull and a 0.3 kg mass: the first lever push (a 2 Nm/deg hold, sdf knob inside the collar) threw the top 141 m.
LATCH = bool(int(os.environ.get("T1_LEVER_LATCH", 1)))
# T1_REST_FIX (default 0, 2026-10-29): resting-contact noise of the TRANSPORT trolley.  Measured (sub-task 10a, after a gripped pull): every body of the empty trolley reads 3-6 cm/s, nearly all vertical, with 0.03 mm
# net motion (body z wanders +-0.01 mm, tire z +-0.02 mm step to step); sleep / stabilization thresholds, solver iterations, linear damping, contact / rest offsets and the slide drive do not change it.  The
# tire and wheel colliders are 'sdf' meshes.  =1: the tire colliders of the transport trolley become convexHull (a tire is convex); =2: every collider mesh of the transport trolley except the handle becomes convexHull; =3: the sdf resolution of the tire / wheel / body colliders is raised to T1_REST_SDF (512); =4: the same for the tire colliders only (the only ones that touch the floor).
REST_FIX = int(os.environ.get("T1_REST_FIX", 0))
LATCH_U_DEG = 35.0; LATCH_LIM = (-6.0, 42.0); LATCH_K, LATCH_D = 0.002, 0.003        # Nm per degree, Nm s per degree.  0.01 turned the knob only 3 deg before the finger rolled the whole trolley north (4 free casters, about 25 N at 1 cm/s): the detent torque has to stay well below the trolley's rolling resistance


# T1_CASTER_REAL=1 (default 0, 2026-10-28): real hospital-caster resistance instead of ROLL_D velocity damping (ROLL_D grows with speed: about 25 N at 1 cm/s and hundreds of N at a 0.3 m/s walking push for one
# 4-caster trolley, the 140 N pinch slipped on the transport trolley, a sideways push yawed it 25-53 deg).  Each tire (roll) joint and each wheel (swivel) joint gets a Coulomb-like friction made of a velocity drive with
# damping D = Tc / w0 and maxForce Tc: it holds a load below Tc at rest (creep w0) and resists with the constant torque Tc once the joint turns faster than w0 (these are plain rigid-body joints, not an articulation, so
# PhysX has no joint-friction attribute; a saturating damper is the same thing).  Tc is read from the stage per trolley: mass = sum over its rigid bodies (MassAPI mass, else density of the physics material x mesh volume),
# load per caster N = m g / (number of rolling casters), Tc_roll = F_ROLL x N x tire radius (the tire radius from its bounding box), Tc_swivel = SWIVEL_ARM x N.
#   F_ROLL 0.03: polyurethane on a hard floor 0.02-0.035 (Holkie), 0.03 (CasterHQ 95A), 0.04-0.08 (Bulldog, concrete); AORN OR-equipment measurements: sustained push 10-37 N for carts, 18-27 N for tower and equipment cart.
#   Breakaway is 1.5-3 x rolling (CasterHQ, Hamilton, AORN initial / sustained 2-3): this friction has ONE level (static = kinetic), so the breakaway here is the rolling value (lower than real, flagged).
#   SWIVEL_ARM 0.001 m: ball-race swivel head, mu 0.01-0.02 on a 25 mm race (EN 12527 limits swivel traction to 20 % of the load, a ceiling not a typical value); no manufacturer number was found (assumption).
CASTER_REAL = bool(int(os.environ.get("T1_CASTER_REAL", 0)))
CASTER_F_ROLL = float(os.environ.get("T1_CASTER_F", 0.03)); CASTER_SWIVEL_ARM = float(os.environ.get("T1_CASTER_SWIVEL_ARM", 0.001))
CASTER_V0 = float(os.environ.get("T1_CASTER_V0", 0.001)); CASTER_W0_SWIVEL = float(os.environ.get("T1_CASTER_W0_SWIVEL", 0.5))      # creep speed under the Tc: 1 mm/s at the tire surface; 0.5 deg/s at the swivel pin


# T1_CASTER_SIZING (2026-10-29): "mass" = Tc from F_ROLL x the PhysX mass of the trolley read after settle (caster_refit), "force" = Tc from the real push force target per trolley (CASTER_FORCE_N, spread over the casters), "usd" = the
# first version (mass from density x mesh volume at author time, 26 / 63 / 85 kg against PhysX 27 / 30 / 29 kg).  caster_refit(app, st) must run once the simulation plays (PhysX has no mass before that); until it runs the USD numbers hold.
CASTER_SIZING = os.environ.get("T1_CASTER_SIZING", "mass")
CASTER_FORCE_N = {"mayo": 8.0, "set_trolley": 19.0, "transport": 25.0}      # real sustained push force (N) per trolley: Mayo ~8, set trolley ~19, transport 10-36 (AORN) ~25
# T1_CASTER_SIZING=spec (2026-10-29): the real trolleys' EMPTY frame mass from product data, plus the payload the simulation already carries (the tray on the transport trolley, the instruments on the Mayo), counted ONCE: payload mass is read from
# PhysX after settle, so the Coulomb torque Tc sees (spec frame + payload) and the payload bodies are not added a second time to anything.  The stage's own trolley mass (PhysX 27 / 30 / 29 kg) is NOT used in this mode.
# Sources (listing figures from search-result summaries, 2026-10-29; the pages themselves answered HTTP 403 to a direct fetch, so a data sheet is still to be read):
#   mayo        14.1 kg = 31 lb  Pedigo double-post Mayo stand, shipping weight 31 / 32 / 40 lb by tray size, 2 in casters (ALIMED listing, https://alimed.com/pedigo-double-post-mayo-stand.html); shipping weight, so an upper bound for the frame.
#                                Others: Drive 43-2923 double post 15.6 lb (7.1 kg); Midmark Mayo stand 19 lb (8.6 kg).  The stage's Mayo is a 0.78 x 0.47 x 0.78 m instrument trolley, larger than a post stand: lower bound.
#   set_trolley 15.5 kg          Fangge B10 stainless instrument trolley, net weight 15.5 kg, 950 x 500 x 920 mm, 100 mm double-brake casters (https://www.medicalexpo.com/prod/fangge-medical/product-124997-1138232.html); the stage's set trolley is 1.06 x 0.58 x 0.89 m.
#                                Another two-shelf instrument table with 125 mm casters: 28 kg (public tender listing, https://storage.mtender.gov.md/get/872eb9c8-cd38-4fe3-865b-f2b5a3bd7896-1596738141555).
#   transport   59.0 kg = 130 lb  Lakeside Deluxe case cart, single door, 42 in high, actual product weight 130 lb, 6 in casters (https://www.universalmedicalinc.com/lakeside-deluxe-case-cart-with-single-door-42-high.html); Pedigo Revolution 29 x 27.5 x 59 in: 117 lb (53 kg).
#                                The stage's transport trolley is 0.71 x 0.51 x 0.94 m with an open tray: a closed case cart is the nearest listed product with a weight.
CASTER_SPEC_KG = {"mayo": 14.1, "set_trolley": 15.5, "transport": 59.0}
_CASTER_PAYLOAD = {}                                                         # trolley name -> [root paths of the objects resting on it in the layout ("on")]
_CASTER_JOINTS = {}                                                          # trolley name -> {"rolls": [(joint path, tire radius)], "swivels": [joint path], "bodies": [paths]}


def _drive(p, kind, k, d, target=None):
    dr = UsdPhysics.DriveAPI.Apply(p, kind)
    dr.CreateTypeAttr("force"); dr.CreateStiffnessAttr(k); dr.CreateDampingAttr(d); dr.CreateMaxForceAttr(1e6)
    if target is not None: dr.CreateTargetPositionAttr(target)


def _coulomb(p, tc, w0):
    """Coulomb-like joint friction: velocity drive, damping tc / w0 (Nm per deg/s), torque capped at tc (Nm); w0 in deg/s"""
    dr = UsdPhysics.DriveAPI.Apply(p, "angular")
    dr.CreateTypeAttr("force"); dr.CreateStiffnessAttr(0.0); dr.CreateDampingAttr(tc / w0); dr.CreateMaxForceAttr(tc)


def _body_mass(b):
    """kg of one rigid body: authored MassAPI mass, else physics-material density (default 1000) x closed-mesh volume (world scale) of the meshes under it"""
    import numpy as np
    from pxr import UsdShade
    if b.HasAPI(UsdPhysics.MassAPI):
        m = UsdPhysics.MassAPI(b).GetMassAttr().Get()
        if m: return float(m)
    tot = 0.0
    for c in Usd.PrimRange(b, Usd.TraverseInstanceProxies()):
        if not c.IsA(UsdGeom.Mesh): continue
        mesh = UsdGeom.Mesh(c); pts = mesh.GetPointsAttr().Get(); fc = mesh.GetFaceVertexCountsAttr().Get(); fi = mesh.GetFaceVertexIndicesAttr().Get()
        if not pts or not fc: continue
        M = UsdGeom.Xformable(c).ComputeLocalToWorldTransform(0); P = np.array([[*M.Transform(v)] for v in pts]); fc = np.asarray(fc); fi = np.asarray(fi)
        start = np.concatenate([[0], np.cumsum(fc)[:-1]]); vol = 0.0
        for k in range(int(fc.max()) - 2):                       # triangle fan k of every face with more than k + 2 corners
            sel = fc > k + 2; i0 = fi[start[sel]]; i1 = fi[start[sel] + k + 1]; i2 = fi[start[sel] + k + 2]
            vol += float(np.einsum("ij,ij->i", P[i0], np.cross(P[i1], P[i2])).sum()) / 6.0
        rho = 0.0
        mat = UsdShade.MaterialBindingAPI(c).ComputeBoundMaterial(materialPurpose="physics")[0]
        if mat: rho = UsdPhysics.MaterialAPI(mat.GetPrim()).GetDensityAttr().Get() or 0.0
        tot += abs(vol) * (rho or 1000.0)
    return tot


def _z_extent(prim):
    """world z extent (m) of the mesh points under a prim (a tire's diameter); 0.1 if there are none"""
    zs = []
    for c in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if c.IsA(UsdGeom.Mesh):
            pts = UsdGeom.Mesh(c).GetPointsAttr().Get(); M = UsdGeom.Xformable(c).ComputeLocalToWorldTransform(0)
            zs += [M.Transform(v)[2] for v in (pts or [])]
    return (max(zs) - min(zs)) if zs else 0.1


def _caster_real(st, top, n):
    """T1_CASTER_REAL: Coulomb-like friction on the roll (tire) and swivel (wheel) joints of one trolley; returns the numbers used"""
    bodies = [p for p in Usd.PrimRange(top, Usd.TraverseInstanceProxies()) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    mass = sum(_body_mass(b) for b in bodies)
    rolls = [p for p in Usd.PrimRange(top) if p.IsA(UsdPhysics.RevoluteJoint) and "tire" in p.GetName().lower()]
    swivels = [p for p in Usd.PrimRange(top) if p.IsA(UsdPhysics.RevoluteJoint) and "wheel" in p.GetName().lower()]
    load = mass * 9.81 / max(1, len(rolls)); bc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]); info = {"mass_kg": round(mass, 2), "casters": len(rolls), "load_per_caster_N": round(load, 1)}
    trs = []
    for p in rolls:
        tgt = UsdPhysics.Joint(p).GetBody1Rel().GetTargets(); r = 0.5 * _z_extent(st.GetPrimAtPath(tgt[0])); tc = CASTER_F_ROLL * load * r; trs.append(r)
        _coulomb(p, tc, math.degrees(CASTER_V0 / r))
    for p in swivels: _coulomb(p, CASTER_SWIVEL_ARM * load, CASTER_W0_SWIVEL)
    _CASTER_JOINTS[top.GetName()] = {"rolls": [(str(p.GetPath()), r) for p, r in zip(rolls, trs)], "swivels": [str(p.GetPath()) for p in swivels], "bodies": [str(p.GetPath()) for p in bodies], "usd_mass_kg": mass}
    if trs: info.update(tire_r_m=round(sum(trs) / len(trs), 4), roll_tc_Nm=round(CASTER_F_ROLL * load * sum(trs) / len(trs), 4), swivel_tc_Nm=round(CASTER_SWIVEL_ARM * load, 4), rolling_force_N=round(CASTER_F_ROLL * mass * 9.81, 1))
    n["caster"] += len(rolls) + len(swivels); n["caster_real"] = info


def _physx_masses(app, paths):
    """PhysX mass (kg) of each rigid body path, with its colliders, from the property query; needs a playing simulation (a few Kit updates)"""
    import omni.usd, omni.physx
    from omni.physx.bindings._physx import PhysxPropertyQueryMode, PhysxPropertyQueryResult
    from pxr import PhysicsSchemaTools, Sdf
    pq = omni.physx.get_physx_property_query_interface(); sid = omni.usd.get_context().get_stage_id(); got = {}
    for p in paths:
        def cb(resp, p=p):
            if resp.result == PhysxPropertyQueryResult.VALID: got[p] = resp.mass
        pq.query_prim(stage_id=sid, prim_id=PhysicsSchemaTools.sdfPathToInt(Sdf.Path(p)), query_mode=PhysxPropertyQueryMode.QUERY_RIGID_BODY_WITH_COLLIDERS, rigid_body_fn=cb, timeout_ms=0)
    for _ in range(40):
        app.update()
        if len(got) == len(paths): break
    return got


def caster_refit(app, st, sizing=None, f_roll=None, v0=None, w0_swivel=None, swivel_arm=None):
    """T1_CASTER_REAL, once the simulation plays: size the friction of every trolley's casters from the PhysX masses (not the USD-volume estimate) and rewrite the drives.  Returns the numbers used per trolley.
    Keyword arguments override the module defaults for this call (probes sweep them in one process); a no-op without the flag or before apply() found any caster."""
    if not CASTER_REAL: return {}
    arm = CASTER_SWIVEL_ARM if swivel_arm is None else swivel_arm
    sizing = sizing or CASTER_SIZING; f_roll = CASTER_F_ROLL if f_roll is None else f_roll; v0 = CASTER_V0 if v0 is None else v0; w0s = CASTER_W0_SWIVEL if w0_swivel is None else w0_swivel
    out = {}
    for name, J in _CASTER_JOINTS.items():
        n = max(1, len(J["rolls"])); got = _physx_masses(app, J["bodies"]); mx = sum(got.values()) if len(got) == len(J["bodies"]) else None
        m = J["usd_mass_kg"] if (sizing == "usd" or mx is None) else mx
        pay = None; pay_ok = True
        if sizing == "spec" and name in CASTER_SPEC_KG:               # spec frame + the payload PhysX carries (counted once)
            pb = [str(q.GetPath()) for rp in _CASTER_PAYLOAD.get(name, []) for q in Usd.PrimRange(st.GetPrimAtPath(rp), Usd.TraverseInstanceProxies()) if q.HasAPI(UsdPhysics.RigidBodyAPI)]
            pg = _physx_masses(app, pb) if pb else {}; pay = sum(pg.values()) if len(pg) == len(pb) else None
            pay_ok = pay is not None; pay = pay if pay_ok else sum(pg.values()); m = CASTER_SPEC_KG[name] + pay          # a payload body PhysX did not answer for counts 0 (flagged in "sizing"), never the stage mass
        load = m * 9.81 / n; info = {"sizing": (sizing + ("" if (sizing != "spec" or pay_ok) else "(payload partial)")) if mx is not None or sizing == "usd" else "usd(no PhysX mass)", "mass_kg": round(m, 2), "physx_kg": None if mx is None else round(mx, 2), "usd_kg": round(J["usd_mass_kg"], 2), "load_per_caster_N": round(load, 1), "payload_kg": None if pay is None else round(pay, 3)}
        tcs = []
        if sizing == "rolld":                                   # diagnosis: the old velocity damping on every caster joint, through the same refit path
            for jp, r in J["rolls"]: _drive(st.GetPrimAtPath(jp), "angular", 0.0, ROLL_D)
            for jp in J["swivels"]: _drive(st.GetPrimAtPath(jp), "angular", 0.0, ROLL_D)
            info.update(rolling_force_N=None, v0_m_s=None); out[name] = info; continue
        for jp, r in J["rolls"]:
            tc = (CASTER_FORCE_N.get(name, f_roll * m * 9.81) * r / n) if sizing == "force" else f_roll * load * r
            _coulomb(st.GetPrimAtPath(jp), tc, math.degrees(v0 / r)); tcs.append(tc / r * n)
        for jp in J["swivels"]: _coulomb(st.GetPrimAtPath(jp), arm * load, w0s)
        info.update(rolling_force_N=round(sum(tcs) / max(1, len(tcs)), 2), swivel_tc_Nm=round(arm * load, 4), v0_m_s=v0); out[name] = info
    import json; print("EV CASTER_REFIT", json.dumps(out), flush=True)
    return out


def tool_phys(st, layout, mode, root="/World/Room"):
    """T1_TOOL_PHYS (default off; 2026-10-30 forceps-jump fix, coordinator variants):  A = maxDepenetrationVelocity 0.1 m/s on every instrument body + 32 solver position iterations on the forceps
    (rigid bodies, and the articulation if one is authored);  B = A + the forceps hinge drive damping x10 (stiffness unchanged).  Runs before the physics parse."""
    from pxr import PhysxSchema
    out = {}
    for o in layout["objects"]:
        if o.get("fixed") or not o["asset"].startswith("instrument_"): continue
        top = st.GetPrimAtPath(f"{root}/{o['name']}"); is_f = o["name"].startswith("forceps"); n = {"bodies": 0, "articulations": 0, "hinges": 0}
        for p in Usd.PrimRange(top):
            if p.HasAPI(UsdPhysics.RigidBodyAPI):
                api = PhysxSchema.PhysxRigidBodyAPI.Apply(p); api.CreateMaxDepenetrationVelocityAttr(0.1); n["bodies"] += 1
                if is_f: api.CreateSolverPositionIterationCountAttr(32)
            if is_f and p.HasAPI(UsdPhysics.ArticulationRootAPI):
                PhysxSchema.PhysxArticulationAPI.Apply(p).CreateSolverPositionIterationCountAttr(32); n["articulations"] += 1
            if mode.upper() == "B" and is_f and p.IsA(UsdPhysics.RevoluteJoint):
                _drive(p, "angular", TOOL_K, TOOL_D * 10.0, 0.0); n["hinges"] += 1
        out[o["name"]] = n
    return out


def tool_post(st, names, mode, root="/World/Room"):
    """T1_TOOL_POST (default off; 2026-10-30 forceps-jump fix, variants E/F/S): physics properties set at RUN time on the named instrument bodies, after sub-task 4 ends (the grasp in 1-4 is untouched).
    Letters in mode: E = maxDepenetrationVelocity 0.1 m/s + 32 position iterations; F = sleepThreshold 0.01 + linear and angular damping 5 (at-rest damping);
    S = (scissors, set just before their release) the same cap as E.  Returns a read-back of the attributes so the log shows what the stage holds."""
    from pxr import PhysxSchema
    out = {}
    for nm in names:
        top = st.GetPrimAtPath(f"{root}/{nm}"); n = 0; rb = None
        for p in Usd.PrimRange(top):
            if not p.HasAPI(UsdPhysics.RigidBodyAPI): continue
            api = PhysxSchema.PhysxRigidBodyAPI.Apply(p)
            if "E" in mode or "S" in mode:
                api.CreateMaxDepenetrationVelocityAttr(0.1); api.CreateSolverPositionIterationCountAttr(32)
            if "F" in mode:
                api.CreateSleepThresholdAttr(0.01); api.CreateLinearDampingAttr(5.0); api.CreateAngularDampingAttr(5.0)
            n += 1; rb = api
        out[nm] = dict(bodies=n, maxDepen=None if rb is None else rb.GetMaxDepenetrationVelocityAttr().Get(), sleepThr=None if rb is None else rb.GetSleepThresholdAttr().Get(),
                       linDamp=None if rb is None else rb.GetLinearDampingAttr().Get())
    return out


def apply(st, layout, root="/World/Room"):
    fixed = {o["name"] for o in layout["objects"] if o.get("fixed")}
    out = {}
    _CASTER_PAYLOAD.clear()
    for o in layout["objects"]:
        if o.get("on"): _CASTER_PAYLOAD.setdefault(o["on"], []).append(f"{root}/{o['name']}")
    for o in layout["objects"]:
        if o["name"] in fixed: continue
        top = st.GetPrimAtPath(f"{root}/{o['name']}"); tool = o["asset"].startswith("instrument_")
        for p in Usd.PrimRange(top):
            if p.IsInstance(): p.SetInstanceable(False)          # instance proxies cannot be edited
        n = {"slide": 0, "caster": 0, "tool_hinge": 0, "hold": 0, "collider": 0}
        for p in Usd.PrimRange(top):
            name = p.GetName().lower()
            if p.IsA(UsdPhysics.PrismaticJoint):
                _drive(p, "linear", LIN_K, LIN_D, 0.0); n["slide"] += 1
            elif p.IsA(UsdPhysics.RevoluteJoint):
                if LATCH and o["name"] == "transport" and "handle" in name:
                    j = UsdPhysics.RevoluteJoint(p); j.CreateLowerLimitAttr(LATCH_LIM[0]); j.CreateUpperLimitAttr(LATCH_LIM[1]); _drive(p, "angular", LATCH_K, LATCH_D, 0.0); n["lever_latch"] = n.get("lever_latch", 0) + 1
                elif "wheel" in name or "tire" in name:
                    if not CASTER_REAL: _drive(p, "angular", 0.0, ROLL_D); n["caster"] += 1          # CASTER_REAL: Coulomb friction set after the loop (_caster_real)
                elif tool:
                    _drive(p, "angular", TOOL_K, TOOL_D, 0.0); n["tool_hinge"] += 1
                else:
                    _drive(p, "angular", HOLD_K, HOLD_D, 0.0); n["hold"] += 1
            elif LATCH and o["name"] == "transport" and p.HasAPI(UsdPhysics.CollisionAPI) and p.IsA(UsdGeom.Mesh) and "handle" in p.GetName().lower():
                UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr("convexHull"); n["collider"] += 1
            elif REST_FIX in (1, 2) and o["name"] == "transport" and p.HasAPI(UsdPhysics.CollisionAPI) and p.IsA(UsdGeom.Mesh) and ("tire" in name or (REST_FIX >= 2 and "handle" not in name)):
                UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr("convexHull"); n["rest_fix"] = n.get("rest_fix", 0) + 1
            elif REST_FIX in (3, 4) and o["name"] == "transport" and p.HasAPI(UsdPhysics.CollisionAPI) and p.IsA(UsdGeom.Mesh) and ("tire" in name or (REST_FIX == 3 and ("wheel" in name or "body" in name))):
                from pxr import PhysxSchema
                sd = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(p); old_r = sd.GetSdfResolutionAttr().Get(); sd.CreateSdfResolutionAttr(int(os.environ.get("T1_REST_SDF", 512))); n["rest_fix"] = n.get("rest_fix", 0) + 1; n["sdf_old"] = old_r
            elif tool and p.HasAPI(UsdPhysics.CollisionAPI) and p.IsA(UsdGeom.Mesh):
                UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr("convexDecomposition"); n["collider"] += 1
        if CASTER_REAL and n["caster"] == 0 and any(x.IsA(UsdPhysics.RevoluteJoint) and "tire" in x.GetName().lower() for x in Usd.PrimRange(top)): _caster_real(st, top, n)
        if LATCH and o["name"] == "transport":
            rb = {p.GetName(): p for p in Usd.PrimRange(top) if p.HasAPI(UsdPhysics.RigidBodyAPI)}; hb = next(v for k, v in rb.items() if k.startswith("handle_01"))
            UsdPhysics.MassAPI.Apply(hb).CreateMassAttr(0.3); fp = UsdPhysics.FilteredPairsAPI.Apply(hb)
            for k, v in rb.items():
                if k.startswith(("body_01", "table_01", "lock_01")): fp.CreateFilteredPairsRel().AddTarget(v.GetPath())
            n["lever_filter"] = 1
        out[o["name"]] = {k: v for k, v in n.items() if v}
    return out
