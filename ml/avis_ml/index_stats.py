"""Print index coverage: tracks, hours and active seconds per category/dataset/split."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .data_index import data_root


def main():
    p = Path(data_root()) / "index.jsonl"
    rows = [json.loads(line) for line in open(p, encoding="utf-8")]
    by = defaultdict(lambda: {"n": 0, "tracks": set(), "hours": 0.0, "active_s": 0, "dur": 0.0})
    for r in rows:
        k = (r["dataset"], r["split"], r["category"])
        e = by[k]
        e["n"] += 1
        e["tracks"].add(r["track_id"])
        e["dur"] += r["duration"]
        e["hours"] += r["duration"] / 3600.0
        e["active_s"] += len(r["active_segments"])
    print(f"rows={len(rows)}")
    print(f"{'dataset':7} {'split':10} {'category':13} {'stems':>6} {'tracks':>6} {'hours':>7} {'active_s':>9}")
    tot_h = tot_a = 0.0
    for k in sorted(by):
        e = by[k]
        print(f"{k[0]:7} {k[1]:10} {k[2]:13} {e['n']:6d} {len(e['tracks']):6d} {e['hours']:7.1f} {e['active_s']:9d}")
        tot_h += e["hours"]
        tot_a += e["active_s"]
    print(f"total hours={tot_h:.1f} active_seconds={tot_a:.0f} ({tot_a/3600:.1f} h)")
    # test isolation check
    tr = {(r["track_id"]) for r in rows if r["split"] == "train"}
    te = {(r["track_id"]) for r in rows if r["split"] == "test"}
    print(f"train_tracks={len(tr)} test_tracks={len(te)} overlap={len(tr & te)}")


if __name__ == "__main__":
    main()
