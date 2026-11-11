"""behavior-tree JSON writers for the H1 people (omni.behavior.tree 2.0 schema, same nodes as stage 0)"""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
LIBS = [{"name": "omni.behavior.tree.core", "version": "1.1"}, {"name": "omni.anim.behavior.tree", "version": "0.3"}]

def move(name, tgt):
    return {"type": "MoveTo", "source": "omni.anim.behavior.tree", "name": name, "portOverrides": {"target": {"type": "SdfPath", "value": tgt}}}

def dwell(name, look, s):
    return {"type": "LookAt", "source": "omni.anim.behavior.tree", "name": name,
            "modifiers": [{"type": "ForceStatus", "source": "omni.behavior.tree.core", "name": "ForceStatus"},
                          {"type": "Timeout", "source": "omni.behavior.tree.core", "name": "Timeout", "portOverrides": {"duration": float(s)}}],
            "portOverrides": {"target": {"type": "SdfPath", "value": look}}}

def write(fname, children, repeat=True):
    root = {"type": "Sequence", "source": "omni.behavior.tree.core", "name": "Root", "children": children}
    if repeat: root["modifiers"] = [{"type": "Repeat", "source": "omni.behavior.tree.core", "name": "Repeat"}]
    p = os.path.join(HERE, "bt", fname); json.dump({"schemaVersion": "2.0.0", "nodeLibraries": LIBS, "root": root}, open(p, "w"), indent=1)
    return "bt/" + fname

def crosser(dw_w, dw_e, fname="h1_crosser.json"):
    """W (west room, hidden) -> E (room 3, north-west corner) -> W, dwelling dw_e at E and dw_w at W"""
    return write(fname, [move("ToE", "/World/Hosp/Cross_E"), dwell("AtE", "/World/Hosp/Look_E", dw_e),
                         move("ToW", "/World/Hosp/Cross_W"), dwell("AtW", "/World/Hosp/Look_W", dw_w)])

def stand(look, fname):
    return write(fname, [dwell("Stand", look, 1.0e5)], repeat=False)
