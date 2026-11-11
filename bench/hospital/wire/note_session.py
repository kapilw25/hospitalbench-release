"""Note harness (maintainer pick 2026-10-27): every model call is ONE fresh claude -p request; nothing is resumed. It replaces
claude_session.Episode (one conversation that re-sends the whole history; a task-0 call grew to 310k tokens and missed the
cache every few calls).

Each request is one user message, built in this order so the prefix stays cacheable (order set 2026-10-27: what changes
least comes first):
  1. system prompt + tool list + NOTE_RULE (the --system-prompt text; unchanged within a run);
  2. the task text (datasheet + sub-task 1 text) .............................. cache marker 1;
  3. the kept pictures (the call in which each sub-task was last graded, and the call just after each scan latched),
     ordered by call number. A post-latch set is HELD when the scan latches and enters this section only when the
     scanner's own sub-task is graded, together with that graded set (maintainer/coordinator fix 2026-10-27: showing it at the
     latch would tell the model its scan registered, which the old harness never did; the grade tells the outcome anyway).
     So this section changes only at a grading ............................... cache marker 2;
  4. the log: one text block per earlier call line (append-only; event texts such as a grade and the next sub-task's
     text are their own blocks, in order) .................................... cache marker 3 on the last block;
  5. the current status text and the 4 current pictures (never cached: plain input).
claude -p normally adds 3 cache markers of its own (2 on the system prompt, 1 at the end) and the API allows 4, so it runs
with DISABLE_PROMPT_CACHING=1 (it then adds none; the markers in the content pass through unchanged, checked with a request
log 2026-10-27) and this module places the 3 markers above (1 h TTL, so a slow call does not lose the cache).

The model starts its reply with a short note (its own words, kept as the first NOTE_CHARS characters in its log line) and
then ONE JSON tool call, as in claude_session.Episode.

    from note_session import NoteEpisode
    ep = NoteEpisode(system_prompt, tools, log_dir, model="claude-opus-5-5", effort="high")
    ep.set_task(text); ep.add_log(line); ep.keep(key, call, sub_task, pictures)
    ep.hold(key, call, sub_task, pictures, tag); ep.release(tag)    # held sets become kept when `tag` (a sub-task) is graded
    tool, args = ep.step(current_blocks)          # ep.last_note, ep.last_rec (usage, cost) after the call
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path

NOTE_CHARS = 150
CACHE = {"type": "ephemeral", "ttl": "1h"}

NOTE_RULE = (
    "\n\nTool calling (this interface has no native tools): start your reply with one short note for your own log (at most "
    f"{NOTE_CHARS} characters: what you see and what you plan next), then give exactly ONE JSON object, in the form "
    '{"tool": "<tool name>", "arguments": {...}}, where the arguments follow that tool\'s JSON schema below. Never call more '
    "than one tool per reply.\n\n"
    "Every turn is a fresh request: your earlier replies are not shown again. Each turn carries the task text, the pictures "
    "kept from some earlier calls (among them every call in which a sub-task was graded), your log of every earlier call "
    f"(call number, sub-task, the action and its result, and the first {NOTE_CHARS} characters of your note), and then the "
    "current status and the four current pictures.\n\nTools:\n"
)

KEPT_HEADER = "Pictures kept from earlier calls (each set: head camera RGB, arm camera RGB, head depth, arm depth):"
LOG_HEADER = (f"Your log of the earlier calls of this run, one line per call (call number, sub-task, the action and its "
              f"result, the first {NOTE_CHARS} characters of your note):")
LOG_EMPTY = "(no earlier calls yet)"


def parse_reply(text: str):
    """(tool, arguments, note) from a reply 'note line(s) + one JSON object'. The JSON object is the outermost {...} if that
    parses, else the last '{' from which a JSON object with a "tool" key decodes. note = the reply text outside that object,
    whitespace collapsed. (None, None, note) if no tool call was found."""
    text = text or ""
    dec = json.JSONDecoder()
    found = None
    a, b = text.find("{"), text.rfind("}")
    if a >= 0 and b > a:
        try:
            obj = json.loads(text[a:b + 1]); found = (obj, a, b + 1)
        except json.JSONDecodeError:
            pass
    if found is None or not (isinstance(found[0], dict) and isinstance(found[0].get("tool"), str)):
        found = None
        for m in reversed([m.start() for m in re.finditer(r"\{", text)]):
            try:
                obj, end = dec.raw_decode(text, m)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and isinstance(obj.get("tool"), str):
                found = (obj, m, end); break
    if found is None:
        return None, None, " ".join(text.split())
    obj, s, e = found
    note = " ".join((text[:s] + " " + text[e:]).split())
    args = obj.get("arguments", {})
    if not isinstance(args, dict):
        return None, None, note
    return obj["tool"], args, note


LIMIT_WORDS = re.compile(r"usage limit|hit your (?:\w+ )?limit|limit reached|limit .{0,40}resets|resets? (?:at )?\d", re.I)


def usage_limit(msgs, res):
    """dict(resets_at, text) when the claude -p call was refused by the account's usage limit, else None. Signs: a rate_limit_event
    whose status is "rejected", or a failed result with no API time and a limit message ("You've hit your limit ... resets ...")."""
    rej = [m.get("rate_limit_info") or {} for m in msgs if m.get("type") == "rate_limit_event"]
    rej = [i for i in rej if str(i.get("status", "")).startswith("rejected")]
    text = str((res or {}).get("result") or "")
    if rej or (not (res or {}).get("duration_api_ms") and LIMIT_WORDS.search(text)):
        ra = next((i.get("resetsAt") for i in rej if i.get("resetsAt")), None)
        m = re.search(r"\|(\d{10})\b", text)                # older message form: "Claude AI usage limit reached|<reset epoch>"
        if ra is None and m: ra = m.group(1)
        return dict(resets_at=float(ra) if ra else None, text=text or json.dumps(rej[:1])[:300])
    return None


def image_block(jpeg: bytes):
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.b64encode(jpeg).decode()}}


class NoteEpisode:
    """One model run under the note harness. State: the task text, the kept picture sets and the log. step() sends one fresh
    request and logs it to log_dir/claude_cli_calls.jsonl (usage, cost, error, reply, note) and log_dir/requests.jsonl (the
    request as sent: every text block in full, images by sha1, the cache marker positions, the section of each block)."""

    def __init__(self, system_prompt: str, tools: list[dict], log_dir, model: str = "claude-opus-5-5", effort: str = "high",
                 timeout_s: float = 1800, sandbox=None, note_chars: int = NOTE_CHARS):
        self.system = system_prompt + NOTE_RULE + json.dumps(tools, indent=1)
        self.log_dir = Path(log_dir); self.log_dir.mkdir(parents=True, exist_ok=True)
        self.sandbox = Path(sandbox or self.log_dir / "sandbox"); self.sandbox.mkdir(parents=True, exist_ok=True)
        self.model, self.effort, self.timeout_s, self.note_chars = model, effort, timeout_s, note_chars
        self.session = "note-harness (fresh request per call)"
        self.task = None; self.log: list[str] = []; self.kept: dict[str, int] = {}; self.sets: dict[int, dict] = {}
        self.n, self.cost = 0, 0.0
        self.last_text = self.last_note = None; self.last_rec = None; self.last_layout = None
        self.held: dict[str, dict] = {}    # key -> dict(call, sub_task, pictures, tag): recorded, not shown until release(tag)
        self.limit_waits = []              # usage-limit waits: dict(call, start, resets_at, wait_s, text); a refused call is waited out, never a model failure
        for f in ("claude_cli_calls.jsonl", "requests.jsonl"):
            open(self.log_dir / f, "w").close()

    # ------------------------------------------------------------------ state
    def set_task(self, text: str):
        assert self.task is None, "the task text is set once per run (it is the cached prefix)"
        self.task = text

    def add_log(self, text: str):
        """append one block to the log (append-only: earlier blocks are never changed)"""
        self.log.append(str(text))

    def keep(self, key: str, call: int, sub_task, pictures):
        """keep the 4 pictures of `call` under `key` (e.g. 'graded 2', 'scan scanA'); a newer call replaces the key's older
        one. pictures = [(caption, jpeg bytes), ...] exactly as sent in that call."""
        if call not in self.sets:
            self.sets[call] = dict(sub_task=sub_task, pictures=[(c, bytes(j)) for c, j in pictures], keys=set())
        old = self.kept.get(key)
        self.kept[key] = call; self.sets[call]["keys"].add(key)
        if old is not None and old != call:
            self.sets[old]["keys"].discard(key)
            if not self.sets[old]["keys"]:
                del self.sets[old]

    def hold(self, key: str, call: int, sub_task, pictures, tag):
        """record the 4 pictures of `call` under `key` WITHOUT showing them; release(tag) moves them into the kept section.
        A newer hold of the same key replaces the older one (only the latest latch counts)."""
        self.held[key] = dict(call=call, sub_task=sub_task, pictures=[(c, bytes(j)) for c, j in pictures], tag=tag)

    def release(self, tag):
        """move every held set whose tag == tag into the kept section (call this when sub-task `tag` is graded); returns their keys"""
        out = []
        for key in [k for k, h in self.held.items() if h["tag"] == tag]:
            h = self.held.pop(key); self.keep(key, h["call"], h["sub_task"], h["pictures"]); out.append(key)
        return out

    def kept_calls(self):
        return sorted(self.sets)

    # ------------------------------------------------------------------ the request
    def compose(self, current: list[dict]):
        """the content blocks of one request and their layout (section per block, cache marker indices)"""
        assert self.task is not None, "set_task() first"
        blocks, sect, marks = [], [], []
        blocks.append({"type": "text", "text": self.task}); sect.append("task"); marks.append(len(blocks) - 1)
        if self.sets:
            blocks.append({"type": "text", "text": KEPT_HEADER}); sect.append("kept")
            for c in self.kept_calls():
                s = self.sets[c]
                blocks.append({"type": "text", "text": f"Kept pictures from call {c} (sub-task {s['sub_task']}):"}); sect.append("kept")
                for cap, jpg in s["pictures"]:
                    blocks.append({"type": "text", "text": cap}); sect.append("kept")
                    blocks.append(image_block(jpg)); sect.append("kept")
            marks.append(len(blocks) - 1)
        blocks.append({"type": "text", "text": LOG_HEADER}); sect.append("log")
        if not self.log:
            blocks.append({"type": "text", "text": LOG_EMPTY}); sect.append("log")
        for line in self.log:
            blocks.append({"type": "text", "text": line}); sect.append("log")
        marks.append(len(blocks) - 1)
        for b in current:
            blocks.append(b); sect.append("current")
        out = []
        for i, b in enumerate(blocks):
            b = dict(b)
            if i in marks: b["cache_control"] = dict(CACHE)
            out.append(b)
        return out, dict(sections=sect, marks=marks, kept_calls=self.kept_calls(), n_log=len(self.log),
                         kept_keys={c: sorted(self.sets[c]["keys"]) for c in self.kept_calls()}, held={k: h["call"] for k, h in self.held.items()},
                         n_images=sum(1 for b in out if b["type"] == "image"))

    def _args(self):
        return ["claude", "-p", "--safe-mode", "--tools", "", "--strict-mcp-config", "--no-chrome", "--model", self.model,
                "--effort", self.effort, "--system-prompt", self.system, "--input-format", "stream-json",
                "--output-format", "stream-json", "--verbose", "--no-session-persistence"]

    RETRY_SLEEP_S = (30, 120, 300)          # between the 4 attempts (the guard stays "4 tries"; a wider window than Episode's 15/30/60 s,
                                            # after the pilot of 2026-10-27 lost its run to a failure that cleared within minutes)

    def _send(self, blocks):
        """one fresh claude -p call; returns (result message or None, error text). Same guard as Episode: 1800 s wall per
        attempt, 4 attempts. Every failed attempt is logged in full to log_dir/wire_errors.jsonl (result text, API error status,
        stderr), so a failure can be read afterwards."""
        line = json.dumps({"type": "user", "message": {"role": "user", "content": blocks}}) + "\n"
        res, err = None, ""
        env = {**os.environ, "DISABLE_AUTO_COMPACT": "1", "DISABLE_PROMPT_CACHING": "1"}
        attempt = 0
        while attempt < 4:
            t0 = time.time()
            try:
                p = subprocess.run(self._args(), input=line, capture_output=True, text=True, cwd=self.sandbox,
                                   timeout=self.timeout_s, env=env)
            except subprocess.TimeoutExpired:
                err = "timeout"; p = None
            if p is not None:
                msgs = []
                for x in p.stdout.splitlines():
                    if x.strip().startswith("{"):
                        try: msgs.append(json.loads(x))
                        except json.JSONDecodeError: pass
                res = next((m for m in msgs if m.get("type") == "result"), None)
                if res and res.get("subtype") == "success" and not res.get("is_error"):
                    return res, ""
                r = res or {}
                err = json.dumps(dict(result=r.get("result"), subtype=r.get("subtype"), is_error=r.get("is_error"), api_error_status=r.get("api_error_status"),
                                      errors=r.get("errors"), stop_reason=r.get("stop_reason"), duration_api_ms=r.get("duration_api_ms"),
                                      returncode=p.returncode, stderr=(p.stderr or "")[-1500:], no_result=res is None,
                                      events=[str(m.get("type")) + ":" + str(m.get("subtype", "")) for m in msgs][-12:]))
                lim = usage_limit(msgs, r)
                res = None
                if lim is not None:                        # the account's usage limit refused the call: wait for the reset and send it again
                    self._wait_limit(lim, err); continue  # (not counted as one of the 4 tries; a refused call is not a model failure)
            with open(self.log_dir / "wire_errors.jsonl", "a") as f:
                f.write(json.dumps(dict(call=self.n, attempt=attempt + 1, time=time.time(), wall_s=round(time.time() - t0, 1), error=err)) + "\n")
            if attempt < 3: time.sleep(self.RETRY_SLEEP_S[attempt])
            attempt += 1
        return None, err

    LIMIT_RECHECK_S = 1800                      # while waiting at a usage limit, try again at least this often (a limit can lift before its reset)

    def _wait_limit(self, lim, err):
        now = time.time(); resets = lim.get("resets_at")
        wait = min(self.LIMIT_RECHECK_S, max(60.0, resets - now + 90)) if resets else min(self.LIMIT_RECHECK_S, 600.0)
        rec = dict(call=self.n, start=now, resets_at=resets, wait_s=round(wait), text=(lim.get("text") or "")[:300])
        self.limit_waits.append(rec)
        with open(self.log_dir / "wire_errors.jsonl", "a") as f:
            f.write(json.dumps(dict(call=self.n, usage_limit=True, time=now, wait_s=round(wait), resets_at=resets, error=err)) + "\n")
        print(f"USAGE_LIMIT call {self.n}: {rec['text'][:120]!r}; resets at "
              f"{time.strftime('%H:%M:%S', time.gmtime(resets)) + ' UTC' if resets else 'unknown'}; waiting {round(wait)} s, then the same request again", flush=True)
        time.sleep(wait)

    def _request_record(self, blocks, layout):
        rows = []
        for b, s in zip(blocks, layout["sections"]):
            if b["type"] == "text": rows.append(dict(s=s, text=b["text"], cc=("cache_control" in b)))
            else:
                raw = base64.b64decode(b["source"]["data"])
                rows.append(dict(s=s, image=hashlib.sha1(raw).hexdigest()[:12], bytes=len(raw), cc=("cache_control" in b)))
        return rows

    def step(self, current: list[dict]):
        """send one request; return (tool, arguments), or (None, raw_text) if the reply held no JSON tool call"""
        blocks, layout = self.compose(current)
        self.n += 1; self.last_layout = layout
        with open(self.log_dir / "requests.jsonl", "a") as f:
            f.write(json.dumps(dict(call=self.n, marks=layout["marks"], kept_calls=layout["kept_calls"], kept_keys=layout["kept_keys"], held=layout["held"], n_log=layout["n_log"],
                                    n_images=layout["n_images"], blocks=self._request_record(blocks, layout))) + "\n")
        res, err = self._send(blocks)
        cost = float((res or {}).get("total_cost_usd") or 0.0); self.cost += cost
        text = (res or {}).get("result") or ""
        tool, args, note = parse_reply(text) if res else (None, None, "")
        self.last_text, self.last_note = text, note
        self.last_rec = dict(call=self.n, session=None, time=time.time(), model=self.model, effort=self.effort,
                             error=err if res is None else None, result_text=(res or {}).get("result"), note=note,
                             usage=(res or {}).get("usage"), total_cost_usd=cost, n_blocks=len(blocks), n_images=layout["n_images"],
                             kept_calls=layout["kept_calls"], n_log=layout["n_log"], marks=layout["marks"])
        with open(self.log_dir / "claude_cli_calls.jsonl", "a") as f:
            f.write(json.dumps(self.last_rec) + "\n")
        if res is None:
            raise RuntimeError(f"claude -p failed: {err}")
        return (tool, args) if tool else (None, text)


class StubNoteEp(NoteEpisode):
    """NoteEpisode with the model replaced by a script of (tool, args) or raw reply strings: compose() and the logs are the
    real ones, no claude call. seen = the composed requests (blocks, layout)."""

    def __init__(self, system_prompt, tools, log_dir, script, cost=0.03):
        super().__init__(system_prompt, tools, log_dir, model="stub", effort="stub")
        self.script = list(script); self.stub_cost = cost; self.seen = []

    def _send(self, blocks):
        self.seen.append(blocks)
        item = self.script.pop(0) if self.script else ("wait", {"seconds": 20})
        if isinstance(item, str):
            text = item
        else:
            tool, args = item
            text = f"stub note {len(self.seen)}: plan {tool}\n" + json.dumps({"tool": tool, "arguments": args})
        cr = sum(len(b.get("text", "")) // 4 + (300 if b["type"] == "image" else 0) for b in blocks[:max(self.last_layout["marks"]) + 1])
        return dict(type="result", subtype="success", is_error=False, result=text, total_cost_usd=self.stub_cost,
                    usage=dict(input_tokens=1500, cache_read_input_tokens=cr, cache_creation_input_tokens=60, output_tokens=200)), ""
