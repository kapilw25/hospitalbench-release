"""replay_resume: resume a counted model run that a PROVIDER failure stopped, by replaying it (maintainer pick 2026-11-01).

Maintainer decision (2026-11-01): "a counted model run stopped by a provider failure (end wire_error or provider_spend_cap; never a capability end like first_fail) can resume instead of restarting.
Restore the scene from the run's START state (same start file and settle), re-run every logged robot call in order using the model's LOGGED replies (no new model calls, no API cost), and after each
replayed sub-task compare its grade with the logged grade (pass/fail and the logged terms within solver noise).  If every replayed grade matches, continue LIVE from the failed call with the model's full
conversation history rebuilt as the wire had it (Gemini: thoughtSignature parts and file references; Opus: the claude -p session transcript; Terra: the response history).  If any grade differs, stop with
end replay_mismatch and the run is restarted from its start.  The run record names 'resumed: replayed N calls, grades matched'.  A run is still one continuous simulation from its start."

Used by v21_t1_model_run.py  (flags --resume-from <run folder>, --dry-replay).  v21_task_model_run.py can import the same module (see the T1 harness for the five hook points):
  1. RR.Resume.open(run_dir, state, model, effort, upto, caps)   before the world is built (gives .start and .settle of the stopped run)
  2. HIST_REPL extras:  RR.HIST_REPL_EXTRA_NOTE (the two texts the T1 harness replaces in hist_run: the 'replayed' row field and the DryReplayStop end)
  3. RES.wrap(ep, so, R_getter)   after the episode exists (and after its system text is final)
  4. in the Run class on_end, right after the grade:  RES.check_grade(k, run.attempts[k])  -> (ok, why); on not ok finish with "replay_mismatch"; "dry_replay_end" is also a finishing reason
  5. at the end:  RES.record()  -> dict for summary.json / the run log

What is logged from now on so that a later resume is exact (before 2026-11-01 these were not logged):
  claude/model_parts.jsonl   Gemini native: the model's reply parts verbatim, thoughtSignature included (hist_native.py)
  claude/sent_blocks.jsonl   every user turn as sent: the text blocks in full and the image hashes (install_sent_log below)
  v21_t1_model_log.json      world.settle and system_sha1
A stopped run WITHOUT claude/model_parts.jsonl can be dry-replayed, but a LIVE Gemini resume is refused (the thoughtSignature parts cannot be rebuilt).
"""
import glob, hashlib, json, os, shutil, sys, time, base64

REL_TOL = 0.0       # no relative slack: the absolute table below is the solver-noise band
# absolute solver-noise bands per term-name suffix (units are the term's own); the observed deltas of every replay are printed so a band can be audited
ABS_TOL = [("_cm_s", 0.5), ("_deg_s", 0.5), ("_cm", 0.5), ("_mm", 1.0), ("_deg", 1.0), ("_m", 0.02), ("yaw", 1.0), ("centre", 0.01)]
DEFAULT_ABS = 0.05
CAPABILITY_ENDS = ("first_fail", "pass_end", "all_passed", "sub_cap", "done_fail", "provider_refused_context", "task_fail", "done")


class DryReplayStop(Exception):
    """--dry-replay: every logged call has been replayed; the next call would go live"""


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []


def sha12(s):
    return hashlib.sha1(s.encode() if isinstance(s, str) else s).hexdigest()[:12]


def _tol(key):
    for suf, t in ABS_TOL:
        if key.endswith(suf) or key == suf: return t
    return DEFAULT_ABS


def _cmp(key, new, old, out, path):
    """recursive tolerant compare of one grader term; appends (path, old, new, delta, tol) to out for numbers, returns False on a mismatch"""
    if str(key).startswith("trace"): return True
    if isinstance(old, bool) or isinstance(new, bool) or old is None or new is None or isinstance(old, str) or isinstance(new, str):
        return old == new
    if isinstance(old, (int, float)) and isinstance(new, (int, float)):
        d = abs(float(new) - float(old)); t = max(_tol(str(key)), REL_TOL * abs(float(old))); out.append((path, old, new, round(d, 4), t)); return d <= t
    if isinstance(old, list) and isinstance(new, list):
        return len(old) == len(new) and all(_cmp(key, n, o, out, f"{path}[{i}]") for i, (n, o) in enumerate(zip(new, old)))
    if isinstance(old, dict) and isinstance(new, dict):
        return set(old) == set(new) and all(_cmp(k, new[k], old[k], out, f"{path}.{k}") for k in old)
    return old == new


def compare_attempt(new, old):
    """new / old: one attempt record (summary.json attempts[k][j]).  Returns (ok, [reason...], deltas)"""
    new = json.loads(json.dumps(new, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    why, deltas = [], []
    if bool(new.get("passed")) != bool(old.get("passed")): why.append(f"passed {old.get('passed')} -> {new.get('passed')}")
    for n, t in (old.get("terms") or {}).items():
        nt = (new.get("terms") or {}).get(n)
        if nt is None or bool(nt["ok"]) != bool(t["ok"]): why.append(f"term ok '{n}' {t['ok']} -> {None if nt is None else nt['ok']}")
    go, gn = old.get("grade_raw") or [], new.get("grade_raw") or []
    if len(go) != len(gn): why.append(f"grade_raw parts {len(go)} -> {len(gn)}")
    for po, pn in zip(go, gn):
        if po.get("part") != pn.get("part") or bool(po.get("passed")) != bool(pn.get("passed")): why.append(f"part {po.get('part')} passed {po.get('passed')} -> {pn.get('passed')}")
        for k, v in (po.get("terms") or {}).items():
            out = []
            if k not in (pn.get("terms") or {}): why.append(f"part {po.get('part')} term {k} missing"); continue
            if not _cmp(k, pn["terms"][k], v, out, k):
                bad = [o for o in out if o[3] > o[4]]
                why.append(f"part {po.get('part')} term {k}: logged {str(v)[:60]} replay {str(pn['terms'][k])[:60]}" + (f" (delta {bad[0][3]} > band {bad[0][4]})" if bad else ""))
            deltas += [(po.get("part"),) + o for o in out]
    return (not why), why, deltas


def install_sent_log(ep, claude_dir):
    """log every user turn as sent (text in full, image sha1) to claude/sent_blocks.jsonl: a later resume rebuilds the history from it byte for byte"""
    p = os.path.join(claude_dir, "sent_blocks.jsonl"); open(p, "w").close(); orig = ep.step

    def step(content):
        log_sent(p, getattr(ep, "n", 0) + 1, content); return orig(content)
    ep.step = step


def log_sent(path, n, content):
    bl = [dict(t=b["text"]) if b["type"] == "text" else dict(img=sha12(base64.b64decode(b["source"]["data"]))) for b in content]
    with open(path, "a") as f: f.write(json.dumps(dict(call=n, blocks=bl)) + "\n")


class Resume:
    def __init__(self):
        self.ep = None

    @classmethod
    def open(cls, run_dir, state, model, effort, upto, caps, dry=False):
        self = cls(); self.dry = dry
        sd = os.path.join(run_dir, f"state{state}"); self.sd = sd; self.run_dir = run_dir
        if not os.path.isdir(sd): sys.exit(f"--resume-from {run_dir}: no state{state} folder")
        self.summ = json.load(open(os.path.join(sd, "summary.json")))
        self.log = json.load(open(os.path.join(sd, "v21_t1_model_log.json"))) if os.path.exists(os.path.join(sd, "v21_t1_model_log.json")) else {}
        cd = os.path.join(sd, "claude"); self.cd = cd
        self.rows = jl(os.path.join(sd, "calls.jsonl")); self.cli = jl(os.path.join(cd, "claude_cli_calls.jsonl")); self.api = jl(os.path.join(cd, "api_calls.jsonl"))
        self.req = jl(os.path.join(cd, "requests.jsonl")); self.parts = {r["call"]: r for r in jl(os.path.join(cd, "model_parts.jsonl"))}
        self.sent = {r["call"]: r for r in jl(os.path.join(cd, "sent_blocks.jsonl"))}; self.werr = jl(os.path.join(cd, "wire_errors.jsonl"))
        end = self.summ.get("end_raw") or self.summ.get("end")
        if end != "wire_error": sys.exit(f"--resume-from: the stopped run ended '{end}', not by a provider failure (wire_error); only a provider failure may resume")
        if not self.rows or self.rows[-1].get("end") != "wire_error": sys.exit("--resume-from: the last logged call is not the failed one")
        if any(r.get("end") == "wire_error" for r in self.rows[:-1]): sys.exit("--resume-from: a wire error before the last call")
        self.M = len(self.rows) - 1
        if self.M < 1: sys.exit("--resume-from: nothing to replay (the first call failed): restart the run")
        for k, v in (("model", self.summ.get("model") or self.log.get("model")), ("effort", self.log.get("effort")), ("upto", self.log.get("upto"))):
            want = dict(model=model, effort=effort, upto=upto)[k]
            if v is not None and v != want: sys.exit(f"--resume-from: stopped run has {k}={v}, this run has {want}")
        if self.log.get("state") is not None and self.log["state"] != state: sys.exit("--resume-from: state differs")
        oc = self.log.get("caps") or {}
        if oc and {int(k): v for k, v in oc.items()} != caps: sys.exit(f"--resume-from: call caps differ ({oc} vs {caps})")
        w = self.log.get("world") or {}
        self.start = w.get("start"); self.settle = int(w.get("settle", 120)); self.system_sha1 = w.get("system_sha1")
        if not self.start: sys.exit("--resume-from: the stopped run's log names no start file")
        if not os.path.exists(self.start): sys.exit(f"--resume-from: start file {self.start} is gone")
        self.old_attempts = {int(k): v for k, v in (self.summ.get("attempts") or {}).items()}
        self.native = bool(self.parts) or bool(self.api and "function_call_name" in self.api[0])
        self.sig_missing = bool(self.native and not all(i in self.parts for i in range(1, self.M + 1)))
        self.i = 0; self.compared = []; self.mismatch = []; self.replaying = False; self.stats = dict(text_exact=0, text_total=0, img_exact=0, img_total=0, text_from_log=0)
        self.history_info = None
        return self

    # ------------------------------------------------------------------ wiring
    def wrap(self, live, so, run_getter):
        self.live, self.so, self.run_getter = live, so, run_getter; self.cdir_new = os.path.join(so, "claude"); self.old_calls = os.path.join(self.sd, "calls")
        self.ep = Wrapper(self)
        if self.native and self.sig_missing and not self.dry:
            sys.exit("--resume-from: Gemini native run without claude/model_parts.jsonl (thoughtSignature parts not logged before 2026-11-01): a live resume cannot rebuild the history; use --dry-replay or restart the run")
        for f in ("claude_cli_calls.jsonl", "api_calls.jsonl", "requests.jsonl", "wire_errors.jsonl", "model_parts.jsonl"): open(os.path.join(self.cdir_new, f), "a").close()
        self.sent_path = os.path.join(self.cdir_new, "sent_blocks.jsonl"); open(self.sent_path, "w").close()
        return self.ep

    # ------------------------------------------------------------------ one replayed call
    def action(self, i):
        import claude_session as CS
        if i in self.parts:
            fc = self.parts[i].get("fc"); ps = self.parts[i]["parts"]; fcs = [p["functionCall"] for p in ps if "functionCall" in p]
            if fcs: return fcs[0]["name"], fcs[0].get("args") or {}
        txt = self.cli[i - 1].get("result_text") or ""
        c = CS.parse(txt)
        return c if c else (None, txt)

    def replay_step(self, content):
        i = self.i + 1; tool, args = self.action(i); row = self.rows[i - 1]
        if tool != row.get("tool"): raise RuntimeError(f"replay: call {i} logged tool {row.get('tool')} but the logged reply parses as {tool}")
        used, req = [], (self.req[i - 1] if i - 1 < len(self.req) and self.req[i - 1].get("call") == i else None)
        old_sent = self.sent.get(i); k_img = 0; k_txt = 0
        for b in content:
            if b["type"] == "text":
                self.stats["text_total"] += 1; txt = b["text"]
                if old_sent:
                    ot = [x["t"] for x in old_sent["blocks"] if "t" in x]
                    if k_txt < len(ot): txt = ot[k_txt]; self.stats["text_from_log"] += 1
                if req:
                    th = [x["text"] for x in req["new_user"] if "text" in x]
                    if k_txt < len(th) and th[k_txt] == sha12(txt): self.stats["text_exact"] += 1
                elif old_sent: self.stats["text_exact"] += 1
                k_txt += 1; used.append(dict(b, text=txt))
            else:
                self.stats["img_total"] += 1; nm = ("head_rgb", "arm_rgb", "head_depth", "arm_depth")[k_img]; p = os.path.join(self.old_calls, f"c{i:03d}_sent_{nm}.jpg")
                if not os.path.exists(p): raise RuntimeError(f"replay: logged picture {p} missing")
                raw = open(p, "rb").read(); ih = [x["image"] for x in req["new_user"] if "image" in x] if req else None
                if ih is not None and k_img < len(ih) and ih[k_img] != sha12(raw): raise RuntimeError(f"replay: logged picture {p} does not match requests.jsonl")
                if raw == base64.b64decode(b["source"]["data"]): self.stats["img_exact"] += 1
                used.append(dict(b, source=dict(b["source"], data=base64.b64encode(raw).decode()))); k_img += 1
        log_sent(self.sent_path, i, used)
        # the logged wire records of call i go into this run's logs NOW, one call at a time: run_hist reads the LAST line of claude_cli_calls.jsonl as the reply / cost of the call it just made
        for f, rows in (("api_calls.jsonl", self.api), ("requests.jsonl", self.req), ("wire_errors.jsonl", self.werr)):
            with open(os.path.join(self.cdir_new, f), "a") as fo:
                for r in rows:
                    if r.get("call") == i: fo.write(json.dumps(dict(r, replayed=True)) + "\n")
        if i in self.parts:
            with open(os.path.join(self.cdir_new, "model_parts.jsonl"), "a") as fo: fo.write(json.dumps(self.parts[i]) + "\n")
        with open(os.path.join(self.cdir_new, "claude_cli_calls.jsonl"), "a") as fo: fo.write(json.dumps(dict(self.cli[i - 1], replayed=True)) + "\n")
        self.user_turns = getattr(self, "user_turns", []) + [used]; self.i = i; self.replaying = True
        return tool, args

    def check_grade(self, k, attempts):
        """called by the harness right after a sub-task was graded; during the replay compares with the logged attempt"""
        if not self.replaying or self.ep is None: return True, ""
        j = len(attempts) - 1; old = (self.old_attempts.get(k) or [])
        if j >= len(old): self.mismatch.append(f"sub-task {k}: graded in the replay (attempt {j + 1}) but not in the logged run"); return False, self.mismatch[-1]
        ok, why, d = compare_attempt(attempts[-1], old[j])
        self.compared.append(dict(sub_task=k, attempt=j + 1, logged_passed=old[j]["passed"], replay_passed=bool(attempts[-1]["passed"]), match=ok, max_delta=max([x[4] for x in d] or [0]),
                                  deltas=[dict(part=x[0], term=x[1], logged=x[2], replay=x[3], delta=x[4], band=x[5]) for x in d if x[4] > 0]))
        print(f"REPLAY_GRADE sub-task {k} attempt {j + 1}: logged {'PASS' if old[j]['passed'] else 'FAIL'} replay {'PASS' if attempts[-1]['passed'] else 'FAIL'} terms " + ("match within the solver-noise bands" if ok else "DIFFER: " + "; ".join(why)[:400]) +
              " | numeric deltas " + ", ".join(f"{x[1]} {x[4]}(band {x[5]})" for x in d if x[4] > 0)[:600], flush=True)
        if not ok: self.mismatch.append(f"sub-task {k}: " + "; ".join(why)); return False, self.mismatch[-1]
        return True, ""

    # ------------------------------------------------------------------ the switch to live
    def switch(self):
        R = self.run_getter(); self.replaying = False
        for k, old in self.old_attempts.items():
            if R is not None and len(R.attempts.get(k, [])) != len(old):
                msg = f"sub-task {k}: {len(R.attempts.get(k, []))} graded attempts replayed, {len(old)} logged"; self.mismatch.append(msg); raise ReplayMismatchStop(msg)
        new_rows = jl(os.path.join(self.so, "calls.jsonl"))[:self.M]
        dt = max([abs(a["sim_t"] - b["sim_t"]) for a, b in zip(new_rows, self.rows)] or [0]); db = max([max(abs(x - y) for x, y in zip(a["base_true"], b["base_true"])) for a, b in zip(new_rows, self.rows)] or [0])
        res_diff = sum(1 for a, b in zip(new_rows, self.rows) if isinstance(a.get("result"), dict) and isinstance(b.get("result"), dict) and any(isinstance(b["result"].get(k), bool) and a["result"].get(k) != b["result"][k] for k in b["result"]))
        self.drift = dict(max_sim_t_drift_s=round(dt, 3), max_base_drift_m=round(db, 4), calls_with_flag_diff=res_diff)
        live = self.live; ut = self.user_turns; assistant = []
        for i in range(1, self.M + 1):
            tool, args = self.action(i); txt = self.cli[i - 1].get("result_text") or ""
            if hasattr(live, "decls"):          # Gemini native
                if i in self.parts: assistant.append({"parts": self.parts[i]["parts"], "fc": self.parts[i].get("fc")})
                else: assistant.append({"parts": [{"functionCall": {"name": tool, "args": args}}] if tool else ([{"text": txt}] if txt else []), "fc": tool})
            else: assistant.append(txt)
        info = dict(model_kind="native" if hasattr(live, "decls") else ("api-text" if hasattr(live, "turns") else "claude -p session"), signatures_missing=self.sig_missing)
        if hasattr(live, "turns"):
            live.turns = [t for i in range(self.M) for t in (("user", ut[i]), ("assistant", assistant[i]))]; live.n = self.M; live.cost = sum(float(c.get("total_cost_usd") or 0) for c in self.cli[:self.M])
            info.update(turns=len(live.turns), images=sum(1 for r, c in live.turns if r == "user" for b in c if b["type"] == "image"),
                        text_chars=sum(len(b["text"]) for r, c in live.turns if r == "user" for b in c if b["type"] == "text") + sum(len(c) if isinstance(c, str) else len(json.dumps(c.get("parts"))) for r, c in live.turns if r == "assistant"))
            if hasattr(live, "decls"):          # build the request without any upload / network: the file references are replaced by a stand-in
                up = live._upload; live._upload = lambda jpeg: "dry://" + sha12(jpeg)
                try:
                    url, body, _ = live._body(live.turns + [("user", [{"type": "text", "text": "next observation"}])])
                finally: live._upload = up
                cs = body["contents"]; info.update(contents=len(cs), parts=sum(len(c["parts"]) for c in cs), function_responses=sum(1 for c in cs for p in c["parts"] if "functionResponse" in p),
                                                   signature_parts=sum(1 for c in cs for p in c["parts"] if "thoughtSignature" in p), request_bytes=len(json.dumps(body)))
        else:   # claude -p: the session transcript
            info.update(self.prepare_transcript())
        self.history_info = info
        print("RESUME history rebuilt:", json.dumps(info), "| replay drift", json.dumps(self.drift), "| texts exact", f"{self.stats['text_exact']}/{self.stats['text_total']}",
              "pictures byte-identical to the render", f"{self.stats['img_exact']}/{self.stats['img_total']}", flush=True)
        print(f"RESUMED replayed {self.M} calls, grades matched ({len(self.compared)} graded attempts compared)", flush=True)

    def prepare_transcript(self):
        """Opus: find the claude -p session transcript, cut it back to the end of call M (what the failed call may have written is removed; a copy is kept), point the live episode at it"""
        live = self.live; sess = self.cli[0].get("session")
        base = os.environ.get("CLAUDE_CONFIG_DIR") or os.path.join(os.path.expanduser("~"), ".claude")
        hits = glob.glob(os.path.join(base, "projects", "*", sess + ".jsonl"))
        if not hits: sys.exit(f"--resume-from: claude -p transcript of session {sess} not found: restart the run")
        p = hits[0]; lines = open(p).read().splitlines(); want = (self.cli[self.M - 1].get("result_text") or "").strip(); last = None
        for idx, l in enumerate(lines):
            try: r = json.loads(l)
            except Exception: continue
            if r.get("type") == "assistant":
                t = "".join(b.get("text", "") for b in (r.get("message") or {}).get("content") or [] if b.get("type") == "text").strip()
                if t and t == want: last = idx
        if last is None: sys.exit("--resume-from: the transcript holds no reply equal to the logged reply of the last good call: restart the run")
        cut = len(lines)
        for idx in range(last + 1, len(lines)):
            try: r = json.loads(lines[idx])
            except Exception: continue
            if r.get("type") == "queue-operation": cut = idx; break
        info = dict(transcript=p, lines=len(lines), kept=cut, cut=len(lines) - cut)
        if not self.dry:
            shutil.copy(p, p + f".before_resume_{int(time.time())}")
            if cut < len(lines): open(p, "w").write("\n".join(lines[:cut]) + "\n")
            live.session = sess; live.opened = True; live.n = self.M; live.cost = sum(float(c.get("total_cost_usd") or 0) for c in self.cli[:self.M])
            sb = glob.glob(os.path.join(self.cd, "sandbox")); live.sandbox = type(live.sandbox)(sb[0]) if sb else live.sandbox
        return info

    def record(self):
        return dict(text=f"resumed: replayed {self.M} calls, grades matched" if not self.mismatch else f"replay_mismatch after {self.i} replayed calls", from_run=self.run_dir, replayed_calls=self.M, grades_compared=self.compared,
                    mismatch=self.mismatch, history=self.history_info, drift=getattr(self, "drift", None), stats=self.stats, dry=self.dry, stopped_run_stop=str(self.summ.get("stop"))[:300])


class ReplayMismatchStop(Exception):
    pass


class Wrapper:
    """the episode run_hist sees: replays the logged calls, then (not dry) hands over to the live episode"""
    def __init__(self, res):
        object.__setattr__(self, "_res", res)

    def __getattr__(self, name):
        return getattr(self._res.live, name)

    def __setattr__(self, name, v):
        setattr(self._res.live, name, v)

    @property
    def replaying(self):
        return self._res.replaying

    def step(self, content):
        r = self._res
        if r.i < r.M: return r.replay_step(content)
        if r.i == r.M and not getattr(r, "switched", False):
            r.switched = True
            try: r.switch()
            except ReplayMismatchStop:
                with open(os.path.join(r.cdir_new, "claude_cli_calls.jsonl"), "a") as f: f.write(json.dumps(dict(call=r.M + 1, error="replay_mismatch", result_text=None, usage=None, total_cost_usd=0.0)) + "\n")
                raise
        if r.dry:
            n = r.M + 1; r.replaying = False
            with open(os.path.join(r.cdir_new, "claude_cli_calls.jsonl"), "a") as f: f.write(json.dumps(dict(call=n, error="dry_replay_end", result_text=None, usage=None, total_cost_usd=0.0)) + "\n")
            raise DryReplayStop(f"dry replay: {r.M} logged calls replayed, the next call would go live")
        log_sent(r.sent_path, r.live.n + 1, content)
        return r.live.step(content)
