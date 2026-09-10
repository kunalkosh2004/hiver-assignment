"""Interactive console conversation with the retrieval-augmented agent.

Lets you hold a multi-turn support conversation the way the golden harness
frames it: each message advances a session clock, the full prior thread is
passed as conversation context, retrieval is temporally filtered, and the
agent replies deterministically with intent / action / confidence + draft.

Controls at the prompt:
    /reset   start a fresh session (new customer)
    /topk N  change retrieved-evidence depth (default 5)
    /inspect show intent + action diagnostics for the last turn
    /quit    exit

Usage:
    python scripts/agent_chat.py
    AGENT_INTENT=data/retrieval/models/intent_weak python scripts/agent_chat.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent.agent import Agent  # noqa: E402

_TS_RE = r"^\w{3} \w{3} \d{2} \d{2}:\d{2}:\d{2} \+0000 \d{4}$"


def _max_corpus_ts() -> str:
    """Latest well-formed historical message timestamp, same format the golden
    harness uses for its temporal filter (so 'now' sorts after history).
    Corpus rows are noisy, so malformed stamps are ignored."""
    import re

    best = None
    rx = re.compile(_TS_RE)
    with (ROOT / "data" / "retrieval" / "corpus.jsonl").open(errors="ignore") as fh:
        for line in fh:
            i = line.find('"customer_timestamp":')
            if i < 0:
                continue
            start = line.find('"', i + len('"customer_timestamp":') + 1) + 1
            end = line.find('"', start)
            val = line[start:end]
            if rx.match(val) and (best is None or val > best):
                best = val
    return best or "Wed Nov 22 09:24:30 +0000 2017"


def _advance(ts: str, minutes: int) -> str:
    """Monotonically advance the HH:MM:SS field of the Twitter-style stamp."""
    import re

    if not re.match(_TS_RE, ts) or "+0000" not in ts:
        return ts
    plus = ts.index("+0000")
    hms = ts[ts.rindex(" ", 0, plus - 1) + 1:plus]
    h, m, s = (int(x) for x in hms.split(":"))
    total = (h * 60 + m + minutes) % (24 * 60)
    prefix = ts[:ts.rindex(" ", 0, plus - 1) + 1]
    return f"{prefix} {total // 60:02d}:{total % 60:02d}:{s:02d} +0000 {ts[plus + 6:]}"


def main() -> int:
    agent = Agent.load()
    clock = _max_corpus_ts()
    history: list[str] = []
    top_k = 5
    last = None

    print("=" * 62)
    print("AmazonHelp support agent — interactive session")
    print("Type a customer message (one line) and hit Enter.")
    print("Commands: /reset  /topk N  /inspect  /quit")
    print("=" * 62)

    while True:
        try:
            text = input("\ncustomer> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return 0
        if not text:
            continue
        cmd = text.lower()
        if cmd == "/quit":
            return 0
        if cmd == "/reset":
            history.clear()
            clock = _max_corpus_ts()
            last = None
            print("[new session]")
            continue
        if cmd.startswith("/topk"):
            try:
                top_k = max(1, int(text.split()[1]))
                print(f"[top_k={top_k}]")
            except (IndexError, ValueError):
                print("[usage: /topk N]")
            continue
        if cmd == "/inspect":
            if last is None:
                print("[no turn yet]")
            else:
                print(f"  intent {last['intent']} (p={last['intent_prob']})")
                print(f"  action {last['action']} confidence={last['action_confidence']}")
                for reason in last["action_reasons"]:
                    print(f"    - {reason}")
                print(f"  evidence {last['evidence_stats']['n_hits']} hits, "
                  f"{last['evidence_stats']['with_reply']} with reply")
            continue

        clock = _advance(clock, 1)
        history.append(text)
        context = "\n".join(history[:-1]) or None

        out = agent.respond(text, conversation_context=context,
                            created_at=clock, top_k=top_k)
        out["evidence_stats"] = {
            "n_hits": len(out["evidence"]),
            "with_reply": sum(1 for h in out["evidence"]
                              if isinstance(h.get("brand_response"), str) and h["brand_response"].strip()),
        }
        last = out

        print(f"  ├ intent    : {out['intent']} (p={out['intent_prob']})")
        print(f"  ├ action    : {out['action']}  confidence={out['action_confidence']}")
        print(f"  └ draft     : {out['draft']}")
        history.append(out["draft"])


if __name__ == "__main__":
    sys.exit(main())