"""IRA's ensure_navmesh_ready gives the navmesh bake only 100 frames; a headless loop runs those in well under a second and the
hospital bake (10 volumes, 1800 prims) needs longer. Same function, but waits up to 180 s of wall time.
PRE_BAKE: callbacks f(stage) run once, after IRA opened the environment and before the navmesh bake (H1 scene edits that
change walkable space, e.g. door leaves removed and the bed moved, must be in the bake)."""
import time, carb, omni.kit.app
PRE_BAKE = []
_done = {"pre": False}

async def patient_ensure_navmesh_ready(max_s=180.0):
    import omni.anim.navigation.core as nav, omni.usd
    if not _done["pre"]:
        st = omni.usd.get_context().get_stage()
        for f in PRE_BAKE: f(st)
        _done["pre"] = True
        for _ in range(5): await omni.kit.app.get_app().next_update_async()
    inav = nav.acquire_interface()
    if inav.get_navmesh() is not None: return True
    if not inav.start_navmesh_baking():
        carb.log_error("navmesh bake did not start"); return False
    t = time.time()
    while inav.get_navmesh() is None:
        await omni.kit.app.get_app().next_update_async(); time.sleep(0.02)
        if time.time() - t > max_s: carb.log_error("navmesh bake timeout"); return False
    print("navmesh baked in s", round(time.time() - t, 1), flush=True)
    return True

def apply():
    import isaacsim.replicator.agent.core.scene_assembly.environment_loader as e
    import isaacsim.replicator.agent.core.scene_assembly.character_loader as c
    e.ensure_navmesh_ready = patient_ensure_navmesh_ready; c.ensure_navmesh_ready = patient_ensure_navmesh_ready
