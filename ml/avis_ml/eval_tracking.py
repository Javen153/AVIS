"""Tracking quality on scenario WAVs through the C# pipeline.

Metrics (proxy, computed from JSONL objects):
  count_err     mean |objects - expected_mid| over frames
  count_in_range fraction of frames with count inside the expected range
  id_switches   ID changes per second for the expected pitched source
  reid_acc      fraction of re-appearances (gap >= 0.4 s) that reuse an old ID

Usage:
  python -m avis_ml.eval_tracking --model models/v1 --out artifacts/eval/tracking.json
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "runtime" / "src" / "Avis.Cli" / "bin" / "x64" / "Release" / "net10.0" / "avis-cli.exe"
SCEN = ROOT / "testdata" / "scenarios"


def run_cli(wav: Path, model: str, out: Path) -> list[dict]:
    cmd = [str(CLI), "analyze", "--in", str(wav), "--source", "neural",
           "--model", model, "--out", str(out)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line]


def expected(name: str) -> tuple[int, int, tuple[float, float]] | None:
    p = SCEN / f"{name}.expect.json"
    if not p.exists():
        return None
    e = json.loads(p.read_text(encoding="utf-8"))
    win = tuple(e.get("window", [0.0, 1e9]))
    for c in e.get("checks", []):
        if c.get("type") == "count":
            return int(c["min"]), int(c["max"]), win
    return None


def pitch_curve(frames: list[dict]) -> dict:
    """Pick one object per frame whose pitch is nearest to the frame's strongest pitch."""
    ids, labels = [], []
    for fr in frames:
        best = None
        for o in fr["objects"]:
            for p in o.get("pitches", []):
                if p.get("strength", 0) < 0.3:
                    continue
                hz = p.get("hz", 0)
                if hz <= 0:
                    continue
                if best is None or p["strength"] > best[0]:
                    best = (p["strength"], hz, o["id"])
        labels.append(None if best is None else round(4 * math.log2(best[1] / 440.0)))
        ids.append(None if best is None else best[2])
    return {"ids": ids, "labels": labels}


def id_switches(curve: dict) -> float:
    ids, labels = curve["ids"], curve["labels"]
    switches = 0
    prev = None
    for i in range(1, len(labels)):
        if labels[i] is None or labels[i] != labels[i - 1]:
            prev = ids[i] if labels[i] is not None else None
            continue
        if prev is not None and ids[i] is not None and ids[i] != prev:
            switches += 1
        prev = ids[i]
    return switches / max(len(labels) / 100.0, 1e-9)


def reid(curve: dict, gap: int = 40) -> float:
    last: dict[int, int] = {}
    seen: dict[int, set[int]] = {}
    ok = total = 0
    for i, (oid, lab) in enumerate(zip(curve["ids"], curve["labels"])):
        if oid is None or lab is None:
            continue
        if lab in last and i - last[lab] >= gap:
            total += 1
            if oid in seen.get(lab, set()):
                ok += 1
        seen.setdefault(lab, set()).add(oid)
        last[lab] = i
    return ok / total if total else 1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/v1")
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    model = a.model if Path(a.model).is_absolute() else str(ROOT / a.model)
    want = [s.strip().upper() for s in a.only.split(",") if s.strip()]
    rows = []
    with tempfile.TemporaryDirectory() as td:
        for wav in sorted(SCEN.glob("S*.wav")):
            name = wav.stem
            if want and name not in want:
                continue
            exp = expected(name)
            if exp is None:
                continue
            frames = run_cli(wav, model, Path(td) / f"{name}.jsonl")
            t0, t1 = exp[2]
            frames = [f for f in frames if t0 <= f.get("t", 0) <= t1]
            counts = [len(f["objects"]) for f in frames]
            mid = 0.5 * (exp[0] + exp[1])
            in_range = sum(1 for c in counts if exp[0] <= c <= exp[1]) / max(len(counts), 1)
            curve = pitch_curve(frames)
            rows.append({
                "scenario": name,
                "expected": exp,
                "count_err": sum(abs(c - mid) for c in counts) / max(len(counts), 1),
                "count_in_range": round(in_range, 3),
                "id_switches_per_s": round(id_switches(curve), 3),
                "reid_acc": round(reid(curve), 3),
                "frames": len(frames),
            })
            print(json.dumps(rows[-1], ensure_ascii=False))
    mean = {k: round(sum(r[k] for r in rows) / max(len(rows), 1), 3)
            for k in ("count_err", "count_in_range", "id_switches_per_s", "reid_acc")}
    result = {"model": a.model, "mean": mean, "scenarios": rows}
    print("mean:", json.dumps(mean, ensure_ascii=False))
    if a.out:
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
