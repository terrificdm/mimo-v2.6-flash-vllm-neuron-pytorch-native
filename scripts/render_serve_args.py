#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Render config/serve_args.json into the environment and the `vllm serve` command line.

  render_serve_args.py --model <checkpoint dir> [--profile 1m] [--spec mtp3] [--port 8000]
                       [--file config/serve_args.json] [--repo-dir <repo>] [--media-dir <dir>]
                       [--format json|env|argv|shell]

* profile: a key of "profiles" (default: "default_profile").
* spec: "off" or a key of the profile's "speculative.options" (default: the profile's "speculative.default").
  An option may carry "overrides": {"args": {flag: value}, "env": {...}, "drop_neuron_config": [...],
  "neuron_config": {...}} applied on top of the profile.
* "${HF_SNAPSHOT}", "${REPO_DIR}" and "${MEDIA_DIR}" in any string are substituted.
* When the rendered neuron_config sets mimo_expert_placement, the file's sha256 must equal "placement_sha256".
Formats: json (profile, spec, env, argv), env (KEY=VALUE lines), argv (one argument per line),
shell (a copy-and-paste command with the environment in front).
"""
import argparse
import hashlib
import json
import os
import shlex
import sys


def sub(x, m):
    if isinstance(x, str):
        for k, v in m.items():
            x = x.replace("${" + k + "}", v)
        return x
    if isinstance(x, list):
        return [sub(y, m) for y in x]
    if isinstance(x, dict):
        return {k: sub(v, m) for k, v in x.items()}
    return x


def build(doc, profile, spec, model, repo_dir, media_dir, port):
    m = {"HF_SNAPSHOT": model, "REPO_DIR": repo_dir, "MEDIA_DIR": media_dir}
    profile = profile or doc["default_profile"]
    if profile not in doc["profiles"]:
        raise SystemExit(f"unknown profile {profile!r}; choose one of {', '.join(doc['profiles'])}")
    p = sub(doc["profiles"][profile], m)
    sp = p.get("speculative") or {}
    spec = spec or sp.get("default", "off")
    options = sp.get("options") or {}
    if spec != "off" and spec not in options:
        raise SystemExit(f"unknown speculative option {spec!r}; choose one of off, {', '.join(k for k in options if k != 'off')}")
    opt = options.get(spec) if spec != "off" else None
    env = dict(sub(doc.get("env", {}), m))
    env.update(p.get("env", {}))
    args = list(p["args"]) + list(sub(doc.get("common_args", []), m))
    nc = dict(p["neuron_config"])
    ov = (opt or {}).get("overrides", {})
    for k in ov.get("drop_neuron_config", []):
        nc.pop(k, None)
    nc.update(ov.get("neuron_config", {}))
    env.update(ov.get("env", {}))
    for flag, val in ov.get("args", {}).items():
        if flag in args:
            args[args.index(flag) + 1] = str(val)
        else:
            args += [flag, str(val)]
    placement = nc.get("mimo_expert_placement")
    if placement:
        want = doc.get("placement_sha256")
        got = hashlib.sha256(open(placement, "rb").read()).hexdigest() if os.path.exists(placement) else None
        if got is None or (want and got != want):
            raise SystemExit(f"expert placement {placement}: sha256 {got} != placement_sha256 {want}")
    argv = ["vllm", "serve", model] + args
    if opt is not None:
        argv += ["--speculative-config", json.dumps(opt["config"])]
    argv += ["--port", str(port), "--additional-config", json.dumps({"neuron_config": nc})]
    return {"profile": profile, "spec": spec, "env": env, "argv": argv}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default=os.path.join(repo, "config", "serve_args.json"))
    ap.add_argument("--profile", default=None)
    ap.add_argument("--spec", default=None)
    ap.add_argument("--model", required=True, help="checkpoint directory (the Hugging Face snapshot)")
    ap.add_argument("--repo-dir", default=repo)
    ap.add_argument("--media-dir", default=os.environ.get("MEDIA_DIR", os.path.join(repo, "assets")))
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--format", choices=("json", "env", "argv", "shell"), default="json")
    a = ap.parse_args()
    r = build(json.load(open(a.file)), a.profile, a.spec, a.model, a.repo_dir, a.media_dir, a.port)
    if a.format == "json":
        print(json.dumps(r, indent=1))
    elif a.format == "env":
        sys.stdout.write("".join(f"{k}={v}\n" for k, v in r["env"].items()))
    elif a.format == "argv":
        sys.stdout.write("".join(x + "\n" for x in r["argv"]))
    else:
        envs = " \\\n  ".join(f"{k}={shlex.quote(v)}" for k, v in r["env"].items())
        argv = r["argv"]
        lines, i = [shlex.join(argv[:3])], 3
        while i < len(argv):
            if i + 1 < len(argv) and argv[i].startswith("--") and not argv[i + 1].startswith("--"):
                lines.append(f"{argv[i]} {shlex.quote(argv[i + 1])}")
                i += 2
            else:
                lines.append(argv[i])
                i += 1
        print(f"env {envs} \\\n  " + " \\\n  ".join(lines))


if __name__ == "__main__":
    main()
