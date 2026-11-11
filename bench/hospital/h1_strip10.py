"""h1_strip10: the video status strip of h1_strip with TEN chips (h1_strip.py hard-codes five). Same pattern as hist_run10 / hist_run8: h1_strip.py is not edited; its source is executed with the chip count 5 -> 10
(each literal asserted present) into a new module, and install(TS) puts that module in sys.modules['h1_strip'] (h1_opus imports it lazily) with the task's NAMES, PROMPTS and live_terms. No model calls."""
import os, sys, types
_HERE = os.path.dirname(os.path.abspath(__file__))
_src = open(os.path.join(_HERE, "h1_strip.py")).read()
for a, n in (("range(1, 6)", 4), ("cw = (width - 16) // 5", 1)):
    assert _src.count(a) == n, f"h1_strip.py changed: update h1_strip10 ({a})"
_src = _src.replace("range(1, 6)", "range(1, 11)").replace("cw = (width - 16) // 5", "cw = (width - 16) // 10")
S10 = types.ModuleType("h1_strip"); S10.__file__ = os.path.join(_HERE, "h1_strip.py")
exec(compile(_src, S10.__file__ + " (ten chips)", "exec"), S10.__dict__)


def install(TS):
    S10.NAMES.clear(); S10.NAMES.update(TS.NAMES); S10.PROMPTS.clear(); S10.PROMPTS.update(TS.PROMPTS); S10.live_terms = TS.live_terms
    sys.modules["h1_strip"] = S10
    return S10
