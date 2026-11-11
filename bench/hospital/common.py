"""Shared start-up for the hospital scripts: a GPU lock file (one Isaac Sim process at a time) and a headless Isaac Sim 6.1 app."""
import fcntl, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
import tempfile
LOCKP = os.path.join(tempfile.gettempdir(), "hospitalbench_isaac.lock")      # one Isaac Sim process at a time per GPU on this machine
S3 = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1"
HOSP = S3 + "/Isaac/Environments/Hospital/hospital.usd"

def start(w=1280, h=720):
    lock = open(LOCKP, "a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("waiting for", LOCKP, flush=True); fcntl.flock(lock, fcntl.LOCK_EX)
    print("GPU lock held:", LOCKP, flush=True)
    t = time.time()
    assert os.environ.get("OMNI_KIT_ACCEPT_EULA") == "YES", "read the NVIDIA Omniverse EULA, then export OMNI_KIT_ACCEPT_EULA=YES"
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "width": w, "height": h, "renderer": "RayTracedLighting"})
    print("app start s", round(time.time() - t, 1), flush=True)
    return app, lock
