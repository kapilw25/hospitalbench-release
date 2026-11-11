"""One episode = one claude -p session, append-only, so the prompt cache is reused (maintainer plan 2026-10-23: rivals 2-6
within 24 h and under 60% of the weekly limit).

Why: the LIBERO wire (LLM_LIBERO/src/inspect_robots_libero/claude_cli.py) re-sends the whole flattened history as a
new prompt on every call; the image horizon rewrites old turns, so the cached prefix never matches (hit 0.13 over
3,418 calls: 50.5M tokens written to the cache, 7.8M read). Here the first call opens a session and every later call
`--resume`s it with ONLY the new observation, so earlier turns stay byte-identical and are read from the cache.

Differences from native tool calling, to report with any result:
- tools are described in the system prompt; the model replies with one JSON object {"tool": ..., "arguments": {...}};
- earlier observations (images included) stay in the session; there is no image horizon unless the caller sends
  fewer images;
- thinking is not shown to later turns (Claude Code does not replay it).

    from claude_session import Episode
    ep = Episode(system_prompt, tools, log_dir, model="claude-opus-5-5", effort="high")
    tool, args = ep.step([{"type": "text", "text": "..."}, {"type": "image", "source": {...}}])
"""
from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from pathlib import Path

REPLY_RULE = (
    "\n\nTool calling (this interface has no native tools): reply with exactly ONE JSON object and nothing else, "
    'in the form {"tool": "<tool name>", "arguments": {...}}, where the arguments follow that tool\'s JSON schema '
    "below. Never call more than one tool per reply.\n\nTools:\n"
)


def parse(text: str):
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b <= a:
        return None
    try:
        obj = json.loads(text[a:b + 1])
    except json.JSONDecodeError:
        return None
    if isinstance(obj, dict) and isinstance(obj.get("tool"), str) and isinstance(obj.get("arguments", {}), dict):
        return obj["tool"], obj.get("arguments", {})
    return None


class Episode:
    """A stateful claude -p conversation; every call is logged to log_dir/claude_cli_calls.jsonl (usage_report.py reads it)."""

    def __init__(self, system_prompt: str, tools: list[dict], log_dir: Path, model: str = "claude-opus-5-5",
                 effort: str = "high", timeout_s: float = 1800, sandbox: Path | None = None):
        self.system = system_prompt + REPLY_RULE + json.dumps(tools, indent=1)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.sandbox = Path(sandbox or self.log_dir / "sandbox")
        self.sandbox.mkdir(parents=True, exist_ok=True)
        self.model, self.effort, self.timeout_s = model, effort, timeout_s
        self.session = str(uuid.uuid4())
        self.n, self.cost = 0, 0.0
        self.opened = False  # set after the first attempt: a retry must --resume, or claude -p says the id is in use

    def _args(self):
        base = ["claude", "-p", "--safe-mode", "--tools", "", "--strict-mcp-config", "--no-chrome", "--model", self.model,
                "--effort", self.effort, "--system-prompt", self.system, "--input-format", "stream-json",
                "--output-format", "stream-json", "--verbose"]
        return base + (["--session-id", self.session] if self.n == 0 and not self.opened else ["--resume", self.session])

    def step(self, content: list[dict]):
        """Send one observation; return (tool, arguments), or (None, raw_text) if the reply was not one JSON tool call."""
        line = json.dumps({"type": "user", "message": {"role": "user", "content": content}}) + "\n"
        res, err = None, ""
        for attempt in range(4):
            try:
                args = self._args()
                self.opened = True
                p = subprocess.run(args, input=line, capture_output=True, text=True, cwd=self.sandbox,
                                   timeout=self.timeout_s, env={**os.environ, "DISABLE_AUTO_COMPACT": "1"})
            except subprocess.TimeoutExpired:
                err = "timeout"
                continue
            msgs = [json.loads(x) for x in p.stdout.splitlines() if x.strip().startswith("{")]
            res = next((m for m in msgs if m.get("type") == "result"), None)
            if res and res.get("subtype") == "success" and not res.get("is_error"):
                break
            err = (p.stderr or "")[-500:] or json.dumps(res)[:500]
            res = None
            time.sleep(min(120, 15 * 2 ** attempt))
        self.n += 1
        # claude -p reports the session's running total on a resumed session: log this call's share only
        total = float((res or {}).get("total_cost_usd") or 0) if res else self.cost
        cost, self.cost = total - self.cost, total
        with (self.log_dir / "claude_cli_calls.jsonl").open("a") as f:
            f.write(json.dumps({"call": self.n, "session": self.session, "time": time.time(), "model": self.model,
                                "effort": self.effort, "error": err if res is None else None,
                                "result_text": (res or {}).get("result"), "usage": (res or {}).get("usage"),
                                "total_cost_usd": cost}) + "\n")
        if res is None:
            raise RuntimeError(f"claude -p failed: {err}")
        text = res.get("result") or ""
        call = parse(text)
        return call if call else (None, text)
