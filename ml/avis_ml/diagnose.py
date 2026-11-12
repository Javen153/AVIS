"""No-training diagnostics: loss terms, head gradients, pitch AP + threshold sweep.

Usage:
  python -m avis_ml.diagnose --ckpt models/v2.1/ckpt_best.pt --name v2.1
  python -m avis_ml.diagnose --onnx ../models/v1/model.onnx --name v1 --clips 40
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from .frontend import TorchFrontend
from .losses import compute_losses
from .mixer import MixConfig, ProceduralMixer
from .model import HID, AvisNet, count_params
from .validate import (composite, evaluate, head_grad_norms, labels_of, make_val_sets)

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent


def _grad_norms(net, tf, base, cfg, device):
    margin_db = float(cfg.get("dc_margin_db", 3.0))
    hooks = []
    net._head_grads = {}
    for name, head in (("emb", net.head_emb), ("act", net.head_act),
                       ("pitch", net.head_pitch), ("sal", net.head_sal)):
        def _hook(_m, _inp, out, name=name):
            out.retain_grad()
            net._head_grads[name] = out
        hooks.append(head.register_forward_hook(_hook))
    b = base.batch(int(cfg.get("batch_size", 8)))
    lab = labels_of(b, tf, margin_db)
    net.train()
    net.zero_grad(set_to_none=True)
    out = net(lab["feat"])
    ls = compute_losses(out, lab, cfg)
    ls["loss"].backward()
    g = head_grad_norms(net)
    net.zero_grad(set_to_none=True)
    for h in hooks:
        h.remove()
    del net._head_grads
    net.eval()
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--onnx", default="")
    ap.add_argument("--config", default=str(ROOT / "configs" / "v2.yaml"))
    ap.add_argument("--name", default="model")
    ap.add_argument("--clips", type=int, default=120, help="examples per subset")
    ap.add_argument("--dev", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if not a.ckpt and not a.onnx:
        ap.error("pass --ckpt or --onnx")
    import yaml
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    cfg["val_proc_examples"] = a.clips
    cfg["val_real_examples"] = a.clips
    cfg["val_batch"] = min(8, int(cfg.get("val_batch", 8)))
    device = torch.device(a.dev)
    tf = TorchFrontend(REPO / "shared" / "frontend" / "v1" / "weights.bin", device)
    base = ProceduralMixer(MixConfig(seconds=float(cfg.get("seconds", 4.0)),
                                     max_sources=int(cfg.get("max_sources", 8))), device)
    if a.onnx:
        from .eval_real import OnnxFrameModel
        net = OnnxFrameModel(a.onnx)
        hid = 384
        nparam = -1
        gnorm = {}
    else:
        ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        ckcfg = ck.get("config", {})
        cfg.update(ckcfg)
        hid = int(ckcfg.get("hid", 384))
        cfg["hid"] = hid
        net = AvisNet(hid=hid).to(device)
        net.load_state_dict(ck["model"])
        nparam = count_params(net)
        gnorm = _grad_norms(net, tf, base, cfg, device)
    val_sets = make_val_sets(cfg, tf, base)
    val = evaluate(net, tf, val_sets, device, cfg, uw=None, sweep=True, cell_acc=True)
    val["composite"] = composite(val)
    val["grad_norms"] = gnorm
    val["params"] = nparam
    val["hid"] = hid
    val["name"] = a.name
    out = Path(a.out) if a.out else REPO / "artifacts" / "diag" / f"{a.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(val, indent=1), encoding="utf-8")
    best = val.get("pitch_sweep", [])
    print(json.dumps({
        "name": a.name, "params": nparam, "n_examples": val["n_examples"],
        "loss": round(val["loss"], 4), "l_dc": round(val["l_dc"], 4),
        "l_act": round(val["l_act"], 4), "l_pitch": round(val["l_pitch"], 4),
        "l_sal": round(val["l_sal"], 4),
        "pitch_ap": val["pitch_ap"], "pitch_f1@0.5": round(val["pitch_f1"], 4),
        "pitch_f1_best": round(val["pitch_f1_best"], 4),
        "pitch_thr_best": val["pitch_thr_best"],
        "pitch_stats": val["pitch_stats"],
        "act_iou": round(val["act_iou"], 4), "cell_acc": round(val["cell_acc"], 4),
        "emb_cos": round(val["emb_cos"], 4),
        "batch_noise": val["batch_noise"], "grad_norms": gnorm,
        "composite": round(val["composite"], 4),
        "sweep_best_row": max(best, key=lambda r: r["f1"]) if best else None,
    }, indent=1))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
