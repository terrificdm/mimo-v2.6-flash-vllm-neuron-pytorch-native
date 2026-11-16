#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check a downloaded Hugging Face snapshot against weights.lock.json.

  verify_weights.py --snapshot <snapshot dir> [--lock weights.lock.json] [--jobs 8] [--size-only]

Checks the file count, the total size and every file's size and sha256 (--size-only skips the sha256, for a quick
re-check). Files in the snapshot that are not in the lock file also count as a failure (another revision mixed in).
Exit code: 0 everything matches; 3 mismatch; 2 bad argument.
"""
import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor


def sha256_file(path, bufsize=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--lock", default=os.path.join(os.path.dirname(here), "weights.lock.json"))
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--size-only", action="store_true")
    a = ap.parse_args()
    try:
        lock = json.load(open(a.lock))
    except (OSError, ValueError) as e:
        print(f"cannot read {a.lock}: {e}", file=sys.stderr)
        return 2
    snap = a.snapshot
    if not os.path.isdir(snap):
        print(f"no snapshot directory {snap}", file=sys.stderr)
        return 3
    want = {f["path"]: f for f in lock["files"]}
    present = set()
    for root, _, files in os.walk(snap, followlinks=True):
        for n in files:
            rel = os.path.relpath(os.path.join(root, n), snap)
            if not rel.startswith(".cache" + os.sep):
                present.add(rel)
    bad = []
    extra, missing = sorted(present - set(want)), sorted(set(want) - present)
    if extra:
        bad.append(f"files not in the lock file: {extra}")
    if missing:
        bad.append(f"missing files: {missing}")
    total = 0
    for rel in sorted(present & set(want)):
        size = os.path.getsize(os.path.join(snap, rel))
        total += size
        if size != want[rel]["size"]:
            bad.append(f"{rel}: size {size} != {want[rel]['size']}")
    if len(present) != lock["file_count"]:
        bad.append(f"file count {len(present)} != {lock['file_count']}")
    if not missing and total != lock["total_bytes"]:
        bad.append(f"total bytes {total} != {lock['total_bytes']}")
    if not a.size_only:
        todo = sorted(present & set(want))
        done = 0

        def check(rel):
            return rel, sha256_file(os.path.join(snap, rel))

        with ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
            for rel, got in ex.map(check, todo):
                done += 1
                if got != want[rel]["sha256"]:
                    bad.append(f"{rel}: sha256 {got} != {want[rel]['sha256']}")
                if done % 20 == 0 or done == len(todo):
                    print(f"verify_weights: {done}/{len(todo)} files hashed", flush=True)
    if bad:
        print("verify_weights: FAILED", file=sys.stderr)
        for b in bad:
            print("  " + b, file=sys.stderr)
        return 3
    mode = "size only" if a.size_only else "size and sha256"
    print(f"verify_weights: OK, {len(present)} files, {total} bytes ({mode}), revision {lock['revision']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
