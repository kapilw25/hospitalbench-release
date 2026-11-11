"""Note harness for API models (maintainer order 2026-10-27: GPT-6.1 Terra and Gemini 4 Argon on the five hospital tasks, state 0, on the
same note harness as Opus). ApiNoteEpisode is note_session.NoteEpisode with only the wire swapped: the request content, its order, the
note rule, the give_up tool, the kept pictures and the hold/release of post-latch pictures are NoteEpisode's own code (inherited, not
copied). Only _send differs: instead of claude -p it posts ONE fresh request to the OpenAI Responses API or the Gemini API.

Same content as the claude -p request, in the same order:
  system text  = NoteEpisode.system (the note-harness system prompt + NOTE_RULE + the tool list as JSON text)
                 OpenAI: a developer message first in `input`; Gemini: systemInstruction
  user message = NoteEpisode.compose(): task text, kept pictures (caption text + JPEG), one log block per line, status + the 4 current
                 pictures; text blocks -> input_text / {text}, JPEG -> input_image data URL / inlineData, in the same order
Not sent (claude -p artefacts, not ours): Claude Code's own system lines, its environment message and its account context block.
Prompt caching (maintainer order 2026-10-27: every model caches; nothing the model sees changes):
  OpenAI: explicit mode (prompt_cache_options {"mode": "explicit", "ttl": "30m"}) with prompt_cache_breakpoint {"mode": "explicit"} on up to
    4 content blocks: end of the task text; end of the kept pictures (placed on the constant log header right after them, as the docs show
    breakpoints only on input_text); end of the previous call's log (the read point; only when nothing before the log changed); end of
    this call's log (the write point for the next call). Implicit mode would put its one breakpoint at the end of the message, which ends in
    the changing current pictures, so only the developer block would ever be read (measured: about 1.5k tokens per call).
  Gemini 4 Argon: implicit caching only (explicit caching is not supported for Argon). Measured 2026-10-27 (note_api_cachetest.py config,
    pilot 2 calls 30-31): a call reads the prefix it shares with an earlier request with the same thinkingLevel, in 2,048-token chunks
    (18,432 of a ~20,000-token shared prefix), once that prefix is over the 4,096-token minimum. Nothing to add: no warm-up request.
Request formats, model ids, accepted effort values and prices are those verified on 2026-10-27 by bench/hospital/tier_measure.py
(logs/tier_measure/effort_values.json; model lists GET /v1/models and GET /v1beta/models).

Per call: usage (input, cached input, cache writes, output, reasoning) is logged raw to <log_dir>/api_calls.jsonl and mapped onto the
fields note_run reads (cache_read = cached, cache_creation = cache writes, input = uncached input, output = output incl. reasoning,
thinking = reasoning); the cost comes from PRICES. Rate-limit and quota refusals (HTTP 429) and provider-side overloads (HTTP 503 / 529,
and any other 5xx whose message says overloaded, unavailable or to try again later; coordinator decision 2026-10-27: a provider outage is
not a model failure) are waited out and the same request sent again (NoteEpisode.limit_waits, never a model failure, not one of the 4
tries); every other error gets NoteEpisode's rule: 4 tries, 30 / 120 / 300 s apart, 1800 s per try. A provider safety block or an empty
reply is the model's reply (as with Claude).
Keys by name only, from the environment: OPENAI_API_KEY, GEMINI_API_KEY (never printed or logged).

    from note_api_session import ApiNoteEpisode, MODELS
    ep = ApiNoteEpisode(system_prompt, tools, log_dir, model="gpt-6.1-terra", effort="high")   # then exactly as NoteEpisode
"""
from __future__ import annotations

import base64
import datetime
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request

from note_session import NoteEpisode

OPENAI_URL = os.environ.get("NOTE_API_OPENAI_BASE", "https://api.openai.com") + "/v1/responses"
GEMINI_URL = os.environ.get("NOTE_API_GEMINI_BASE", "https://generativelanguage.googleapis.com") + "/v1beta/models/{model}:generateContent"

# model id -> provider, the effort values its API accepts (tested 2026-10-27, logs/tier_measure/effort_values.json), max output tokens
MODELS = {
    "gpt-6.1-terra": dict(provider="openai", efforts=("none", "low", "medium", "high", "xhigh", "max"), max_out=64000, short="terra"),
    "gpt-6.1-sol": dict(provider="openai", efforts=("low", "medium", "high", "xhigh", "max"), max_out=64000, short="sol"),
    "gpt-6-astra": dict(provider="openai", efforts=("low", "medium", "high", "xhigh", "max"), max_out=64000, short="astra"),
    "gemini-4-argon": dict(provider="gemini", efforts=("low", "medium", "high"), max_out=32768, short="argon"),     # outputTokenLimit 32768
}
# USD per 1M tokens, standard tier, read 2026-10-27 (developers.openai.com/api/docs/models/<id>; ai.google.dev/gemini-api/docs/pricing),
# as tier_measure.PRICES. OpenAI: uncached input, cache read, cache write, output (reasoning included). Gemini: implicit caching, no write
# charge; output includes thinking; Argon's rates double from 2026-12-01 (the date of the call decides).
PRICES = {
    "gpt-6.1-terra": [dict(since="2026-01-01", input=1.0, cached=0.05, write=1.25, output=5.0)],
    "gpt-6.1-sol": [dict(since="2026-01-01", input=2.0, cached=0.10, write=2.5, output=10.0)],
    "gpt-6-astra": [dict(since="2026-01-01", input=10.0, cached=1.0, write=12.5, output=50.0)],     # maintainer-pasted price card 2026-11-01
    "gemini-4-argon": [dict(since="2026-01-01", input=2.0, cached=0.10, write=0.0, output=10.0),
                       dict(since="2026-12-01", input=4.0, cached=0.20, write=0.0, output=20.0)],
}


def key(name):
    """an API key by name, from the environment (the value is never printed)"""
    v = os.environ.get(name, "").strip()
    if not v: raise SystemExit(name + " is not set in the environment; no call was made")
    return v


def price(model, when=None):
    """the price row in force on `when` (UTC datetime; default now)"""
    day = (when or datetime.datetime.now(datetime.timezone.utc)).strftime("%Y-%m-%d")
    rows = [p for p in PRICES[model] if p["since"] <= day]
    return dict(rows[-1])


def cost_usd(model, u, when=None):
    """u = mapped usage (uncached, cached, write, output) -> dollars"""
    p = price(model, when)
    return (u["uncached"] * p["input"] + u["cached"] * p["cached"] + u["write"] * p["write"] + u["output"] * p["output"]) / 1e6


def duration_s(txt):
    """'37s', '1.5s', '250ms', '6m0s', '1h2m3s', '20' -> seconds (None if unreadable)"""
    if txt is None: return None
    s = str(txt).strip()
    try: return float(s)
    except ValueError: pass
    tot, ok = 0.0, False
    for v, unit in re.findall(r"([\d.]+)\s*(ms|h|m|s)", s):
        tot += float(v) * {"ms": 0.001, "s": 1, "m": 60, "h": 3600}[unit]; ok = True
    return tot if ok else None


# ------------------------------------------------------------------ the two wires
def to_openai(model, system, blocks, effort, max_out, breakpoints=None):
    """breakpoints: indices of the content blocks that carry an explicit cache breakpoint (None = no caching options: implicit default)"""
    content = []
    for b in blocks:
        if b["type"] == "text": content.append({"type": "input_text", "text": b["text"]})
        else: content.append({"type": "input_image", "image_url": "data:" + b["source"]["media_type"] + ";base64," + b["source"]["data"]})
    body = {"model": model, "input": [{"role": "developer", "content": [{"type": "input_text", "text": system}]}, {"role": "user", "content": content}],
            "reasoning": {"effort": effort}, "max_output_tokens": max_out, "store": False}
    if breakpoints is not None:
        for i in breakpoints: content[i]["prompt_cache_breakpoint"] = {"mode": "explicit"}
        body["prompt_cache_options"] = {"mode": "explicit", "ttl": "30m"}
    return body


def to_gemini(system, blocks, effort, max_out):
    parts = []
    for b in blocks:
        if b["type"] == "text": parts.append({"text": b["text"]})
        else: parts.append({"inlineData": {"mimeType": b["source"]["media_type"], "data": b["source"]["data"]}})
    return {"systemInstruction": {"parts": [{"text": system}]}, "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"maxOutputTokens": max_out, "thinkingConfig": {"thinkingLevel": effort}}}


def read_openai(d):
    """-> (reply text, usage mapped, extra)"""
    u = d.get("usage") or {}; ind = u.get("input_tokens_details") or {}; outd = u.get("output_tokens_details") or {}
    text, refusal = "", ""
    for it in d.get("output", []) or []:
        if it.get("type") == "message":
            for c in it.get("content", []) or []:
                if c.get("type") == "output_text": text += c.get("text", "")
                if c.get("type") == "refusal": refusal += c.get("refusal", "")
    inp, cached, write = u.get("input_tokens", 0) or 0, ind.get("cached_tokens", 0) or 0, ind.get("cache_write_tokens", 0) or 0
    mu = dict(input=inp, uncached=inp - cached - write, cached=cached, write=write, output=u.get("output_tokens", 0) or 0, reasoning=outd.get("reasoning_tokens", 0) or 0)
    return (text or refusal), mu, dict(status=d.get("status"), incomplete=d.get("incomplete_details"), refusal=refusal or None,
                                       effort_echo=(d.get("reasoning") or {}).get("effort"), usage_raw=u, response_id=d.get("id"))


def read_gemini(d):
    u = d.get("usageMetadata") or {}
    cand = (d.get("candidates") or [{}])[0]
    text = "".join(p.get("text", "") for p in ((cand.get("content") or {}).get("parts") or []) if not p.get("thought"))
    inp, cached = u.get("promptTokenCount", 0) or 0, u.get("cachedContentTokenCount", 0) or 0
    out, th = u.get("candidatesTokenCount", 0) or 0, u.get("thoughtsTokenCount", 0) or 0
    mu = dict(input=inp, uncached=inp - cached, cached=cached, write=0, output=out + th, reasoning=th)
    fr = cand.get("finishReason")
    return text, mu, dict(status=fr, incomplete=None if fr in ("STOP", None) else fr, refusal=(d.get("promptFeedback") or {}).get("blockReason"),
                          effort_echo=None, usage_raw=u, response_id=d.get("responseId"))


OVERLOAD_WORDS = re.compile(r"overload|try again later|retry later|try again in|temporarily|unavailable|high demand|capacity", re.I)


def limit_refusal(provider, http, d, headers):
    """dict(wait_s, kind, text) when the provider refused the call for a rate limit or quota (HTTP 429) or is overloaded (HTTP 503 / 529, or
    another 5xx whose message says overloaded / unavailable / try again later), else None"""
    err = ((d or {}).get("error") or {}) if isinstance((d or {}).get("error"), dict) else {}
    text = str(err.get("message") or json.dumps(d)[:300])
    overload = http in (503, 529) or (500 <= http < 600 and (OVERLOAD_WORDS.search(text) is not None or err.get("status") == "UNAVAILABLE"))
    if http != 429 and not overload: return None
    if overload:
        kind = f"overload http {http} " + str(err.get("status") or err.get("code") or err.get("type") or "")
        w = duration_s(headers.get("retry-after"))
        if w is None: w = next((duration_s(x.get("retryDelay")) for x in (err.get("details") or []) if "RetryInfo" in str(x.get("@type"))), None)
        return dict(wait_s=w, kind=kind.strip(), text=text)
    if provider == "openai":
        kind = err.get("code") or err.get("type") or "rate_limit"
        w = duration_s(headers.get("retry-after"))
        if w is None:
            ws = [duration_s(headers.get(h)) for h in ("x-ratelimit-reset-requests", "x-ratelimit-reset-tokens")]
            ws = [x for x in ws if x is not None]; w = max(ws) if ws else None
    else:
        kind = err.get("status") or "RESOURCE_EXHAUSTED"
        w = next((duration_s(x.get("retryDelay")) for x in (err.get("details") or []) if "RetryInfo" in str(x.get("@type"))), None)
        if w is None: w = duration_s(headers.get("retry-after"))
    return dict(wait_s=w, kind=str(kind), text=text)


def post(url, body, headers, timeout):
    """-> (http status, parsed json, response headers lower-cased)"""
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", **headers}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read()), {k.lower(): v for k, v in r.headers.items()}
    except urllib.error.HTTPError as e:
        txt = e.read().decode(errors="replace"); hd = {k.lower(): v for k, v in (e.headers or {}).items()}
        try: return e.code, json.loads(txt), hd
        except json.JSONDecodeError: return e.code, {"raw": txt[:2000]}, hd


class ApiNoteEpisode(NoteEpisode):
    """NoteEpisode on the OpenAI Responses API or the Gemini API (one fresh request per call; content and order from NoteEpisode.compose)"""
    LIMIT_MIN_WAIT_S = 5.0

    def __init__(self, system_prompt, tools, log_dir, model, effort, timeout_s: float = 1800, sandbox=None, openai_explicit_cache: bool = True):
        if model not in MODELS: raise SystemExit(f"unknown API model {model!r}; known: {sorted(MODELS)}")
        spec = MODELS[model]
        if effort not in spec["efforts"]: raise SystemExit(f"effort {effort!r} not accepted by {model}; accepted: {spec['efforts']}")
        super().__init__(system_prompt, tools, log_dir, model=model, effort=effort, timeout_s=timeout_s, sandbox=sandbox)
        self.provider, self.max_out = spec["provider"], spec["max_out"]
        self.session = f"note-harness API ({self.provider}, fresh request per call)"
        self.limit_streak = 0
        self.openai_explicit_cache = openai_explicit_cache
        self._prev_log_count, self._prev_stable_sig, self.last_breakpoints = 0, None, None
        open(self.log_dir / "api_calls.jsonl", "w").close()

    def _openai_breakpoints(self, blocks):
        """content-block indices for OpenAI's explicit breakpoints (at most 4): task end, kept end (on the log header), previous log end, log end"""
        from note_session import LOG_HEADER
        marks = [i for i, b in enumerate(blocks) if "cache_control" in b]
        h = next(i for i, b in enumerate(blocks) if b["type"] == "text" and b["text"] == LOG_HEADER)
        cur = marks[-1]; n_log = len(self.log)
        stable = hashlib.sha1(json.dumps([b.get("text") or b["source"]["data"] for b in blocks[:h + 1]]).encode()).hexdigest()
        bps = [0]
        if len(marks) == 3: bps.append(h)                                  # kept pictures present: their end = just before the constant log header
        if stable == self._prev_stable_sig and 1 <= self._prev_log_count < n_log:
            bps.append(h + self._prev_log_count)                           # where the previous call wrote its log end: the read point
        bps.append(cur)
        self._prev_log_count, self._prev_stable_sig = n_log, stable
        return sorted(set(bps))

    def _args(self):                     # never used: there is no claude -p here
        raise RuntimeError("ApiNoteEpisode sends over HTTP, not claude -p")

    def _body(self, blocks, breakpoints=None):
        if self.provider == "openai":
            return OPENAI_URL, to_openai(self.model, self.system, blocks, self.effort, self.max_out, breakpoints), {"Authorization": "Bearer " + key("OPENAI_API_KEY")}
        return GEMINI_URL.format(model=self.model), to_gemini(self.system, blocks, self.effort, self.max_out), {"x-goog-api-key": key("GEMINI_API_KEY")}

    def _api_log(self, **kw):
        with open(self.log_dir / "api_calls.jsonl", "a") as f:
            f.write(json.dumps(dict(call=self.n, time=time.time(), provider=self.provider, model=self.model, effort=self.effort, **kw)) + "\n")

    def _wait_api_limit(self, lim, err):
        """wait out a rate-limit / quota refusal (recorded like a usage-limit wait), then the same request goes again"""
        self.limit_streak += 1
        w = lim["wait_s"]
        if w is None: w = min(self.LIMIT_RECHECK_S, 30.0 * 2 ** min(6, self.limit_streak - 1))      # no hint: 30, 60, 120 ... s, then every 30 min
        wait = min(self.LIMIT_RECHECK_S, max(self.LIMIT_MIN_WAIT_S, w * 1.1 + 2))
        now = time.time()
        rec = dict(call=self.n, start=now, resets_at=now + wait, wait_s=round(wait, 1), text=f"{lim['kind']}: {lim['text']}"[:300], provider=self.provider)
        self.limit_waits.append(rec)
        with open(self.log_dir / "wire_errors.jsonl", "a") as f:
            f.write(json.dumps(dict(call=self.n, usage_limit=True, time=now, wait_s=round(wait, 1), kind=lim["kind"], error=err)) + "\n")
        print(f"RATE_LIMIT call {self.n} ({self.provider} {lim['kind']}): {lim['text'][:120]!r}; waiting {round(wait)} s, then the same request again", flush=True)
        time.sleep(wait)

    def _send(self, blocks):
        """one fresh API request; returns (a claude-shaped result dict or None, error text) for NoteEpisode.step"""
        bps = self._openai_breakpoints(blocks) if (self.provider == "openai" and self.openai_explicit_cache) else None
        self.last_breakpoints = bps
        url, body, headers = self._body([{k: v for k, v in b.items() if k != "cache_control"} for b in blocks], bps)
        res, err = None, ""
        attempt = 0
        while attempt < 4:
            t0 = time.time(); http, d, hd = -1, None, {}
            try:
                http, d, hd = post(url, body, headers, self.timeout_s)
            except Exception as e:                                        # network error, timeout
                err = f"{type(e).__name__}: {e}"[:800]
            wall = round(time.time() - t0, 1)
            if http == 200:
                text, mu, extra = read_openai(d) if self.provider == "openai" else read_gemini(d)
                c = cost_usd(self.model, mu); self.limit_streak = 0
                self._api_log(attempt=attempt + 1, http=200, wall_s=wall, usage=mu, cost_usd=c, price=price(self.model), reply_chars=len(text), breakpoints=bps, **extra)
                return dict(type="result", subtype="success", is_error=False, result=text, total_cost_usd=c, duration_api_ms=int(wall * 1000),
                            usage=dict(input_tokens=mu["uncached"], cache_read_input_tokens=mu["cached"], cache_creation_input_tokens=mu["write"],
                                       output_tokens=mu["output"], output_tokens_details=dict(thinking_tokens=mu["reasoning"]),
                                       provider=self.provider, input_total=mu["input"])), ""
            if http != -1: err = json.dumps(dict(http=http, error=(d or {}).get("error", d)))[:1500]
            lim = limit_refusal(self.provider, http, d, hd) if http != -1 else None
            self._api_log(attempt=attempt + 1, http=http, wall_s=wall, error=err, rate_limit=lim)
            if lim is not None:
                self._wait_api_limit(lim, err); continue                  # not one of the 4 tries; a refused call is not a model failure
            with open(self.log_dir / "wire_errors.jsonl", "a") as f:
                f.write(json.dumps(dict(call=self.n, attempt=attempt + 1, time=time.time(), wall_s=wall, error=err)) + "\n")
            if attempt < 3: time.sleep(self.RETRY_SLEEP_S[attempt])
            attempt += 1
        return None, err


def make_episode(model, effort, system_prompt, tools, log_dir):
    """None for the Claude path (run_note then builds its own NoteEpisode, byte-identical to note_t1/t3/h1.py), else an ApiNoteEpisode"""
    if model.startswith("claude-"):
        return None
    return ApiNoteEpisode(system_prompt, tools, log_dir, model=model, effort=effort)
