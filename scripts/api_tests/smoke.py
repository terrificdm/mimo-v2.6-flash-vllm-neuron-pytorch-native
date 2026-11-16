#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Functional smoke test of a running server (standard library only).

Cases (greedy unless noted; thinking off unless noted):
  models     /v1/models lists the served model
  chat_en    chat, English question with a one-word answer
  chat_zh    chat, Chinese question
  math       chat, arithmetic
  complete   /v1/completions
  stream     streamed chat: content deltas, a finish reason and [DONE]
  think      thinking on (sampled): the reply has reasoning and/or content
  tools      tool calling with tool_choice auto: a get_weather call with a JSON "city" argument
  json       structured output with a JSON schema: the content parses and has the field
  conc4      4 chat requests at the same time, all correct
  image / video / audio   base64 media from assets/ (skipped with --no-mm, e.g. on the 8k profiles)

Usage: python3 scripts/api_tests/smoke.py [--no-mm] [--out <prefix>]
Environment: MIMO_API_BASE (default http://127.0.0.1:8000), MIMO_API_MODEL (default mimo-v2.6-flash).
Exit code: 0 every case passed; 1 some case failed; 2 server unreachable.
"""
import argparse
import concurrent.futures as cf
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from ask import data_url  # noqa: E402

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "assets")
OFF = {"enable_thinking": False}


def chat(content, max_tokens=64, **kw):
    kw.setdefault("chat_template_kwargs", OFF)
    return C.post("/v1/chat/completions", C.chat_body(content, max_tokens=max_tokens, **kw))


def text_of(rec):
    return (C.message(rec).get("content") or "").strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="output path prefix (writes <out>.json and <out>.md)")
    ap.add_argument("--no-mm", action="store_true", help="skip the image / video / audio cases")
    a = ap.parse_args()
    if not C.server_ready():
        print(f"server not reachable at {C.base_url()}", file=sys.stderr)
        return 2
    s = C.Suite("smoke", a.out)

    def check(cid, desc, rec, ok, why):
        s.add(cid, desc, "pass" if ok and rec.get("http_status") == 200 else "fail",
              f"HTTP {rec.get('http_status')} {why}", records=[rec])

    with urllib.request.urlopen(C.base_url() + "/v1/models", timeout=10) as r:
        ids = [m["id"] for m in json.load(r)["data"]]
    s.add("models", "/v1/models", "pass" if C.MODEL in ids else "fail", f"models {ids}")

    for cid, q, want in [("chat_en", "What is the capital of France? Answer with one word.", "Paris"),
                         ("chat_zh", "中国的首都是哪座城市？只回答城市名。", "北京"),
                         ("math", "What is 17 * 23? Answer with the number only.", "391")]:
        rec = chat(q)
        check(cid, q, rec, want in text_of(rec), f"answer {text_of(rec)[:80]!r}, expected {want!r}")

    rec = C.post("/v1/completions", {"model": C.MODEL, "prompt": "The capital city of Japan is", "max_tokens": 16,
                                     "temperature": 0})
    txt = (C.choice0(rec).get("text") or "")
    check("complete", "/v1/completions", rec, "Tokyo" in txt, f"text {txt[:80]!r}")

    rec = C.post_stream("/v1/chat/completions", C.chat_body("Count from 1 to 5, separated by commas.", max_tokens=64,
                                                            stream=True, chat_template_kwargs=OFF))
    content, fin = C.stream_concat(rec.get("chunks") or [], "content"), C.stream_finish(rec.get("chunks") or [])
    check("stream", "streamed chat", rec, bool(rec.get("done")) and fin is not None and "5" in content,
          f"content {content[:60]!r}, finish {fin}, done {rec.get('done')}")

    rec = C.post("/v1/chat/completions", C.chat_body("What is 12 + 30?", max_tokens=1024, temperature=1.0, top_p=0.95))
    msg = C.message(rec)
    reasoning, content = C.reasoning_of(msg) or "", msg.get("content") or ""
    check("think", "thinking on", rec, bool(reasoning or content),
          f"reasoning {len(reasoning)} chars, content {content.strip()[:60]!r}")

    tools = [{"type": "function", "function": {
        "name": "get_weather", "description": "Get the current weather of a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
    rec = chat("What is the weather in Paris right now? Use the tool.", max_tokens=256, tools=tools, tool_choice="auto")
    calls = C.message(rec).get("tool_calls") or []
    try:
        args = json.loads(calls[0]["function"]["arguments"]) if calls else {}
    except (ValueError, KeyError, TypeError):
        args = {}
    check("tools", "tool call", rec, bool(calls) and calls[0]["function"]["name"] == "get_weather" and "city" in args,
          f"tool_calls {[c['function']['name'] for c in calls]} args {args}")

    schema = {"type": "object", "properties": {"city": {"type": "string"}, "country": {"type": "string"}},
              "required": ["city", "country"], "additionalProperties": False}
    rec = chat("Name the capital of Italy and its country.", max_tokens=128,
               response_format={"type": "json_schema", "json_schema": {"name": "capital", "schema": schema}})
    try:
        obj = json.loads(text_of(rec))
    except ValueError:
        obj = None
    check("json", "structured output (JSON schema)", rec, isinstance(obj, dict) and "city" in obj and "country" in obj,
          f"content {text_of(rec)[:80]!r}")

    qs = [("What is 2 + 2? Answer with the number only.", "4"), ("What is 3 * 7? Answer with the number only.", "21"),
          ("What is the capital of Germany? One word.", "Berlin"), ("日本的首都是哪座城市？只回答城市名。", "东京")]
    with cf.ThreadPoolExecutor(4) as ex:
        recs = list(ex.map(lambda q: chat(q[0]), qs))
    ok = all(r.get("http_status") == 200 and w in text_of(r) for r, (_, w) in zip(recs, qs))
    s.add("conc4", "4 concurrent chats", "pass" if ok else "fail",
          "answers " + ", ".join(repr(text_of(r)[:20]) for r in recs), records=recs)

    if not a.no_mm:
        media = [("image", "image_url", "number_739.png", "What number is written in the image? Answer with the number only.",
                  ["739"]),
                 ("video", "video_url", "digits_1234.mp4",
                  "Which digits appear in the video, in order? Answer with the digits separated by commas.",
                  ["1", "2", "3", "4"]),
                 ("audio", "audio_url", "audio/jfk.wav", "Transcribe this audio exactly.", ["ask not"])]
        for cid, kind, f, q, want in media:
            content = [{"type": kind, kind: {"url": data_url(os.path.join(ASSETS, f))}}, {"type": "text", "text": q}]
            rec = chat([{"role": "user", "content": content}], max_tokens=64)
            ans = text_of(rec)
            check(cid, f"{cid} {f}", rec, all(w in ans.lower() for w in want), f"answer {ans[:80]!r}")

    c = s.counts()
    print(f"smoke: {c['pass']} pass, {c['fail']} fail")
    return s.exit_code()


if __name__ == "__main__":
    sys.exit(main())
