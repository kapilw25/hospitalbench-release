"""wire_fireworks: Muse Spark Open (Fireworks serverless, OpenAI-compatible chat API) as a session of the HospitalBench model runner.

Written 2026-11-16 for maintainer button Q239 ("Argon + one open model", step 2: Gemini Argon and Muse Spark Open, one run per chore and start).
A WIRE changes only how a model is called (R4): the task text, tools, pictures and graders are the runner's own.

Same contract as hist_session.HistApiEpisode / hist_native.NativeGeminiEpisode, so v21_task_model_run.py, hist_run.py, hist_session.finish_summary and
hospital_skills.install_last_pic use it unchanged:
  ep.step(content) -> (tool, args) or (None, text)         content = the runner's blocks ({"type":"text"} / {"type":"image","source":{base64 jpeg}})
  ep.turns             list of ("user", blocks) / ("assistant", dict)  (install_last_pic reads and edits the user turns)
  ep.n, ep.cost, ep.model, ep.effort, ep.session, ep.limit_waits, ep.provider_refused, ep.flex, ep.flex_wait_s, ep.flex_fallback, ep.decls
Tools: the SAME name, description and input_schema of every other model's tool list, declared natively (tools=[{"type":"function","function":{name,description,parameters}}]);
no tool JSON and no REPLY_RULE in the system text (as the Argon wire).  parallel_tool_calls is False; if a reply still carries several calls only the first is executed and stored.
History: every earlier user turn (text and pictures, as data: URLs) and every earlier assistant turn is re-sent in order; a tool message ("the action ran; its result is in the
status text below", the Argon wire's own words) follows each call before the next user turn.  The model's reasoning_content is sent back with its own turn
(REASONING_HISTORY = "preserved", the server's default, echoed in the Fireworks-Conversation-Options header and logged per call); "dropped" leaves it out.
Per call (the runner's own formats): claude/claude_cli_calls.jsonl (usage -> input_tokens, cache_read_input_tokens, output_tokens, thinking_tokens; total_cost_usd), claude/api_calls.jsonl
(raw usage, price row, wall_s, finish_reason, the reasoning effort the server echoed), requests.jsonl, wire_errors.jsonl.
Dollars: PRICE (USD per 1M tokens, bench_models.PLANNED musespark, read 2026-11-15): uncached input, cached input, output (reasoning included in completion_tokens).
Retries, always logged with the cause, never silent:
  429 and 503/529/overload  -> waited out (Retry-After, else 30 s doubling to 30 min), same request again, not a try, never a model failure (RATE_LIMIT line, wire_errors usage_limit)
  timeout (600 s)           -> same request again, up to 8 times without using a try (RETRY_HUNG)
  other 5xx, network error  -> 4 tries, 30 / 120 / 300 s apart (RETRY_5XX / RETRY_NET)
  empty reply, finish_reason not stop -> same request again, up to 3 times (RETRY_EMPTY), then wire_error; finish_reason stop with no call = the model's own empty action (counts)
  401/402/403, 412 billing  -> fatal (provider_auth / provider_credit_exhausted), nothing retried
  other 4xx                 -> the provider refused this request (a context-window refusal is the model's own end): recorded verbatim, run ends (provider_refused), no retry
No fallback to another model, tier or format: a call that cannot be answered ends the run with the cause.
The key (FIREWORKS_API_KEY, read from the environment only) is sent only to HOST and never printed or logged."""
import json, os, re, time, hashlib, urllib.request, urllib.error, urllib.parse
from pathlib import Path

HOST = "api.fireworks.ai"
URL = os.environ.get("WIRE_FIREWORKS_URL", "https://api.fireworks.ai/inference/v1/chat/completions")     # tests point it at a local stub; the host check below still applies to the key
MODELS = {"accounts/fireworks/models/muse-spark-open": dict(efforts=("low", "medium", "high", "xhigh", "max"), max_out=32768, short="musespark", context=1048576)}
# efforts: the values the API accepted on 2026-11-16 (a probe: high and low echoed back in Fireworks-Conversation-Options; "minimal", "medium", "xhigh", "max", "none", "adaptive" are listed by its validation error; the server default is xhigh)
PRICE = dict(inp=1.25, cached=0.15, out=4.25)           # USD per 1M tokens; bench_models.PLANNED["musespark"]["price"] (selftest in test_wire_fireworks.py compares them)
REASONING_HISTORY = "preserved"                          # "preserved" (server default) = the model's reasoning_content goes back with its own turn; "dropped" = left out
REQUEST_TIMEOUT_S = 600.0
HUNG_RETRIES = 8
RETRY_SLEEP_S = (30, 120, 300)
LIMIT_RECHECK_S = 1800
LIMIT_MIN_WAIT_S = 5.0
TOOL_ACK = "the action ran; its result is in the status text below"       # the Argon wire's functionResponse text (hist_native.py)
OVERLOAD_WORDS = re.compile(r"overload|try again later|retry later|try again in|temporarily|unavailable|high demand|capacity", re.I)


def key():
    """FIREWORKS_API_KEY from the environment only (the value is never printed or logged)."""
    v = os.environ.get("FIREWORKS_API_KEY", "").strip()
    if not v: raise SystemExit("FIREWORKS_API_KEY is not set in the environment")
    return v


def post(url, body, headers, timeout):
    """-> (http status, parsed json, response headers lower-cased).  Refuses to send the key anywhere but HOST (the tests send to a local stub WITHOUT a key)."""
    if "Authorization" in headers and urllib.parse.urlparse(url).hostname != HOST: raise RuntimeError("refusing to send the Fireworks key to a host other than " + HOST)
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", **headers}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r: return r.status, json.loads(r.read()), {k.lower(): v for k, v in r.headers.items()}
    except urllib.error.HTTPError as e:
        txt = e.read().decode(errors="replace"); hd = {k.lower(): v for k, v in (e.headers or {}).items()}
        try: return e.code, json.loads(txt), hd
        except json.JSONDecodeError: return e.code, {"raw": txt[:2000]}, hd


def cost_usd(mu):
    return (mu["uncached"] * PRICE["inp"] + mu["cached"] * PRICE["cached"] + mu["output"] * PRICE["out"]) / 1e6


def read_chat(d, headers=None):
    """a 200 body -> (text, reasoning, tool_calls, mu, extra).  mu = the runner's usage mapping: input (all prompt tokens), uncached, cached, write (0: no write charge), output (completion tokens, reasoning included), reasoning."""
    ch = (d.get("choices") or [{}])[0]; msg = ch.get("message") or {}
    u = d.get("usage") or {}; hd = headers or {}
    inp = int(u.get("prompt_tokens") or 0)
    cached = int(((u.get("prompt_tokens_details") or {}).get("cached_tokens")) or hd.get("fireworks-cached-prompt-tokens") or 0)
    out = int(u.get("completion_tokens") or 0)
    rs = int(((u.get("completion_tokens_details") or {}).get("reasoning_tokens")) or ((u.get("output_tokens_details") or {}).get("reasoning_tokens")) or 0)
    mu = dict(input=inp, uncached=max(0, inp - cached), cached=cached, write=0, output=out, reasoning=rs)
    try: opts = json.loads(hd.get("fireworks-conversation-options") or "{}")
    except ValueError: opts = {}
    extra = dict(finish_reason=ch.get("finish_reason"), status=ch.get("finish_reason"), usage_raw=u, response_id=d.get("id"), echo_reasoning_effort=opts.get("reasoning_effort"),
                 echo_reasoning_history=opts.get("reasoning_history"), echo_parallel_tool_calls=opts.get("parallel_tool_calls"))
    return (msg.get("content") or ""), (msg.get("reasoning_content") or msg.get("reasoning") or ""), (msg.get("tool_calls") or []), mu, extra


def limit_refusal(http, d, hd):
    """dict(wait_s, kind, text) for a rate limit (429) or an overload (503 / 529 / a 5xx that says overloaded), else None"""
    err = (d or {}).get("error"); err = err if isinstance(err, dict) else {}
    text = str(err.get("message") or json.dumps(d)[:300])
    overload = http in (503, 529) or (500 <= http < 600 and OVERLOAD_WORDS.search(text) is not None)
    if http != 429 and not overload: return None
    w = None
    try: w = float(hd.get("retry-after")) if hd.get("retry-after") else None
    except ValueError: w = None
    return dict(wait_s=w, kind=("overload http %d" % http) if overload else (err.get("code") or err.get("type") or "rate_limit"), text=text)


class FireworksEpisode:
    """Episode-like full-history conversation with a Fireworks model (see the module docstring)"""
    provider = "fireworks"

    def __init__(self, system_prompt, tools, log_dir, model, effort, timeout_s: float = REQUEST_TIMEOUT_S, openai_cache=None, flex: bool = False, reasoning_history: str = None, **kw):
        if model not in MODELS: raise SystemExit(f"unknown Fireworks model {model!r}; known: {sorted(MODELS)}")
        spec = MODELS[model]
        if effort not in spec["efforts"]: raise SystemExit(f"effort {effort!r} not accepted by {model}; accepted: {spec['efforts']}")
        if flex: raise SystemExit("Fireworks serverless has no flex tier: --flex is refused, not ignored")
        assert all(t.get("name") != "give_up" for t in tools), "counted runs have no give_up tool"
        self.reasoning_history = reasoning_history or REASONING_HISTORY
        assert self.reasoning_history in ("preserved", "dropped")
        self.system = system_prompt                                      # no REPLY_RULE, no tool JSON (native tools)
        self.decls = [{"type": "function", "function": dict(name=t["name"], description=t["description"], parameters=t["input_schema"])} for t in tools]
        self.model, self.effort, self.timeout_s, self.max_out = model, effort, min(float(timeout_s), REQUEST_TIMEOUT_S), spec["max_out"]
        self.session = f"full-history API (fireworks {model}, every earlier turn re-sent, reasoning {self.reasoning_history})"
        self.flex, self.flex_wait_s, self.flex_fallback = False, 0.0, None
        self.log_dir = Path(log_dir); self.log_dir.mkdir(parents=True, exist_ok=True)
        self.turns = []                                                  # ("user", blocks) / ("assistant", dict(text, reasoning, tool_call))
        self.n, self.cost = 0, 0.0
        self.limit_waits, self.limit_streak, self.provider_refused = [], 0, None
        for f in ("claude_cli_calls.jsonl", "api_calls.jsonl", "requests.jsonl"): open(self.log_dir / f, "w").close()

    # ------------------------------------------------------------------ request
    def _messages(self, turns):
        msgs = [{"role": "system", "content": self.system}]; prev_id = None
        for role, c in turns:
            if role == "user":
                if prev_id: msgs.append({"role": "tool", "tool_call_id": prev_id, "content": TOOL_ACK}); prev_id = None
                msgs.append({"role": "user", "content": [{"type": "text", "text": b["text"]} if b["type"] == "text" else
                                                         {"type": "image_url", "image_url": {"url": "data:" + b["source"]["media_type"] + ";base64," + b["source"]["data"]}} for b in c]})
            elif c and (c.get("text") or c.get("tool_call")):             # an empty reply has no message to send
                m = {"role": "assistant", "content": c.get("text") or ""}
                if self.reasoning_history == "preserved" and c.get("reasoning"): m["reasoning_content"] = c["reasoning"]
                if c.get("tool_call"): m["tool_calls"] = [c["tool_call"]]; prev_id = c["tool_call"]["id"]
                msgs.append(m)
        return msgs

    def _body(self, turns):
        return URL, {"model": self.model, "messages": self._messages(turns), "tools": self.decls, "tool_choice": "auto", "parallel_tool_calls": False,
                     "max_tokens": self.max_out, "reasoning_effort": self.effort, "stream": False}

    @staticmethod
    def _images(turns): return sum(1 for r, c in turns if r == "user" for b in c if b["type"] == "image")

    def _hdr(self): return {"Authorization": "Bearer " + key()} if urllib.parse.urlparse(URL).hostname == HOST else {}

    # ------------------------------------------------------------------ logs
    def _jl(self, name, row):
        with (self.log_dir / name).open("a") as f: f.write(json.dumps(row) + "\n")

    def _log_call(self, res_text, mu, cost, err):
        usage = None if mu is None else dict(input_tokens=mu["uncached"], cache_read_input_tokens=mu["cached"], cache_creation_input_tokens=mu["write"], output_tokens=mu["output"],
                                             output_tokens_details=dict(thinking_tokens=mu["reasoning"]), input_total=mu["input"])
        self._jl("claude_cli_calls.jsonl", {"call": self.n, "session": self.session, "time": time.time(), "model": self.model, "effort": self.effort, "error": err,
                                            "result_text": res_text, "usage": usage, "total_cost_usd": cost})

    def _wait(self, lim, err):
        self.limit_streak += 1; w = lim["wait_s"]
        if w is None: w = min(LIMIT_RECHECK_S, 30.0 * 2 ** min(6, self.limit_streak - 1))
        wait = min(LIMIT_RECHECK_S, max(LIMIT_MIN_WAIT_S, w * 1.1 + 2)); now = time.time()
        rec = dict(call=self.n + 1, start=now, resets_at=now + wait, wait_s=round(wait, 1), text=f"{lim['kind']}: {lim['text']}"[:300], provider=self.provider)
        self.limit_waits.append(rec); self._jl("wire_errors.jsonl", dict(usage_limit=True, error=err, **rec))
        print(f"RATE_LIMIT call {self.n + 1} (fireworks {lim['kind']}): {lim['text'][:120]!r}; waiting {round(wait)} s, then the same request again", flush=True)
        time.sleep(wait)

    # ------------------------------------------------------------------ one call
    def step(self, content):
        turns = self.turns + [("user", content)]; n_img = self._images(turns)
        url, body = self._body(turns)
        self._jl("requests.jsonl", dict(call=self.n + 1, turns=len(turns), images=n_img, breakpoints=None,
                                        new_user=[dict(text=hashlib.sha1(b["text"].encode()).hexdigest()[:12], len=len(b["text"])) if b["type"] == "text" else
                                                  dict(image=hashlib.sha1(b["source"]["data"].encode()).hexdigest()[:12]) for b in content]))
        hdr = self._hdr(); err = ""; attempt = 0; hung = 0
        while attempt < 4:
            t0 = time.time(); http, d, hd = -1, None, {}
            try: http, d, hd = post(url, body, hdr, self.timeout_s)
            except Exception as e: err = f"{type(e).__name__}: {e}"[:800]
            wall = round(time.time() - t0, 1)
            if http == 200:
                text, reasoning, tcs, mu, extra = read_chat(d, hd); c = cost_usd(mu); self.limit_streak = 0
                fr = extra["finish_reason"]
                if not text.strip() and not tcs and fr != "stop":             # EMPTY REPLY RULE (as the Gemini wire): stop = the model's own empty action; anything else is provider-side
                    self._jl("wire_errors.jsonl", dict(call=self.n + 1, attempt=attempt + 1, time=time.time(), wall_s=wall, empty_reply=True, finish_reason=fr, cost_usd=c, retried=attempt < 3))
                    self.cost += c; print(f"RETRY_EMPTY call {self.n + 1} attempt {attempt + 1}/4: empty reply, finish_reason={fr}: " + ("sending the same request again" if attempt < 3 else "giving up: wire_error"), flush=True)
                    if attempt < 3: attempt += 1; continue
                    self.n += 1; self._log_call(None, None, 0.0, f"provider empty reply: finish_reason={fr}")
                    raise RuntimeError(f"fireworks provider empty reply: finish_reason={fr}")
                call, stored_call, bad = None, None, None
                if tcs:
                    fn = (tcs[0].get("function") or {}); raw = fn.get("arguments")
                    try: args = json.loads(raw) if (raw or "").strip() else {}
                    except ValueError: args = None
                    if fn.get("name") and isinstance(args, dict): call = (fn["name"], args); stored_call = dict(id=tcs[0].get("id") or "call_0", type="function", function=dict(name=fn["name"], arguments=json.dumps(args)))
                    else: bad = f"tool call {fn.get('name')!r} with unreadable arguments: {str(raw)[:300]}"       # the model's own malformed call: counts as no valid call
                self.n += 1; self.cost += c
                self._jl("api_calls.jsonl", dict(call=self.n, time=time.time(), provider=self.provider, model=self.model, effort=self.effort, attempt=attempt + 1, http=200, wall_s=wall, usage=mu,
                                                 cost_usd=c, price=dict(PRICE), cache_mode="automatic_prefix", flex_requested=False, service_tier="serverless", turns=len(turns), reply_chars=len(text),
                                                 thoughts_tokens=mu["reasoning"], function_calls=len(tcs), function_call_name=(call[0] if call else None), malformed=bad, reasoning_chars=len(reasoning), **extra))
                self._log_call(json.dumps({"tool": call[0], "arguments": call[1]}) if call else (text or bad or ""), mu, c, None)
                self.turns = turns + [("assistant", dict(text=(text + ("\n[" + bad + "]" if bad else "")) if not call else text, reasoning=reasoning, tool_call=stored_call))]
                return call if call else (None, text or bad or "")
            if http != -1: err = json.dumps(dict(http=http, error=(d or {}).get("error", d)))[:1500]
            low = err.lower()
            if http in (401, 403): raise RuntimeError(f"provider_auth: Fireworks refused the key (HTTP {http}); the maintainer must check FIREWORKS_API_KEY: " + err[:300])
            if http == 402 or (http in (412, 429) and any(w in low for w in ("suspended", "billing", "credit", "spending"))):
                raise RuntimeError("provider_credit_exhausted: the Fireworks account is out of credit or over its limit; the maintainer must add credits: " + err[:300])
            lim = limit_refusal(http, d, hd) if http != -1 else None
            if lim is not None: self._wait(lim, err); continue
            if 400 <= http < 500:                                          # the provider refused this request: recorded verbatim, no retry (a context-window refusal is the model's own end)
                self.n += 1; self.provider_refused = dict(call=self.n, http=http, turns=len(turns), images=n_img, response=d)
                self._log_call(None, None, 0.0, "provider_refused: " + json.dumps(dict(http=http, response=d)))
                self._jl("wire_errors.jsonl", dict(provider_refused=True, time=time.time(), **self.provider_refused))
                print("PROVIDER_REFUSED", http, json.dumps(d)[:300], flush=True)
                raise RuntimeError("provider_refused: " + json.dumps(dict(http=http, response=d)))
            timed_out = "imed out" in err or "Timeout" in err
            self._jl("wire_errors.jsonl", dict(call=self.n + 1, attempt=attempt + 1, time=time.time(), wall_s=wall, error=err, **({"retry_timeout": True} if timed_out else {})))
            if timed_out and hung < HUNG_RETRIES:
                hung += 1; print(f"RETRY_HUNG call {self.n + 1} hung try {hung}/{HUNG_RETRIES} after {wall} s (fireworks): sending the same request again (does not use a try)", flush=True); continue
            print(f"RETRY_{'NET' if http == -1 else '5XX'} call {self.n + 1} attempt {attempt + 1}/4 ({err[:120]}): " + ("sending the same request again" if attempt < 3 else "giving up: wire_error"), flush=True)
            if attempt < 3: time.sleep(RETRY_SLEEP_S[attempt])
            attempt += 1
        self.n += 1; self._log_call(None, None, 0.0, err)
        raise RuntimeError(f"fireworks API failed: {err}")
