"""Compare live vs EMA weights on the fixed validation set (diagnostic only)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .frontend import TorchFrontend
from .mixer import MixConfig, ProceduralMixer
from .model import AvisNet
from .validate import composite, evaluate, make_val_sets

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {})
    hid = int(cfg.get("hid", 384))
    tf = TorchFrontend(ROOT.parent / "shared" / "frontend" / "v1" / "weights.bin", dev)
    base = ProceduralMixer(MixConfig(seconds=float(cfg.get("seconds", 4.0)),
                                     max_sources=int(cfg.get("max_sources", 8))), device=dev)
    sets = make_val_sets(cfg, tf, base)
    net = AvisNet(hid=hid).to(dev)
    out = {}
    for name in ("model", "ema"):
        net.load_state_dict(ck[name])
        v = evaluate(net, tf, sets, dev, cfg, uw=None, sweep=False, cell_acc=False)
        out[name] = {"composite": composite(v), "pitch_ap": v["pitch_ap"],
                     "pitch_f1": v["pitch_f1"], "act_iou": v["act_iou"],
                     "pitch_frac": v["pitch_stats"]["frac_gt_0.5"],
                     "pitch_mean": v["pitch_stats"]["mean"]}
    print(json.dumps({"step": ck.get("step"), **out}, indent=1))


if __name__ == "__main__":
    main()
