"""cell_acc / pitch F1 / act IoU on real held-out clips (MUSDB test, Slakh test).

Usage:
  python -m avis_ml.eval_real --ckpt models/v2/ckpt_best.pt --clips 6 --out artifacts/eval/real.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .eval import eval_batch
from .metrics import pitch_ap, pitch_f1_sweep
from .real_eval import (index_rows, mixdown, musdb_clip, slakh_clip, slakh_track_dirs,
                        targets, tracks)
from .frontend import TorchFrontend
from .model import AvisNet

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
SKIP_EMPTY_DB = -45.0


def _active_stems(stems: dict, min_db: float = SKIP_EMPTY_DB) -> list[str]:
    keep = []
    for k, v in stems.items():
        sr = v[0]
        rms = float(np.sqrt(np.mean(sr.astype(np.float64) ** 2)) + 1e-12)
        if 20 * np.log10(rms) > min_db:
            keep.append(k)
    return keep


class OnnxFrameModel:
    """Runs the exported step model frame by frame (v1 baseline without a checkpoint)."""

    def __init__(self, path: str):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        self.sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
        shape = self.sess.get_inputs()[1].shape
        hid = int(shape[2]) if len(shape) == 3 and isinstance(shape[2], int) else 384
        self.h = np.zeros((2, 1, hid), dtype=np.float32)

    def __call__(self, feat):
        x = feat.detach().float().cpu().numpy()  # [B,T,3,192]
        B, T = x.shape[0], x.shape[1]
        emb, act, pit, sal = [], [], [], []
        for b in range(B):
            self.h = np.zeros_like(self.h)
            eb, ab, pb, sb = [], [], [], []
            for i in range(T):
                e, a, p, s, self.h = self.sess.run(None, {"feat": x[b, i:i + 1], "h_in": self.h})
                eb.append(e[0]); ab.append(a[0]); pb.append(p[0]); sb.append(s[0])
            emb.append(np.stack(eb)); act.append(np.stack(ab))
            pit.append(np.stack(pb)); sal.append(np.stack(sb))
        return {"emb": torch.from_numpy(np.stack(emb)).to(feat.device),
                "act": torch.from_numpy(np.stack(act)).to(feat.device),
                "pitch": torch.from_numpy(np.stack(pit)).to(feat.device),
                "sal": torch.from_numpy(np.stack(sal)).to(feat.device)}

    def eval(self):
        return self

    def train(self, mode=True):
        return self


def run_clips(net, tf, clips: list[tuple[str, str, dict, str]], dev):
    acc = {}
    details = []
    with torch.no_grad():
        for kind, name, stems, start in clips:
            keys = _active_stems(stems)
            if len(keys) < 2:
                continue
            mix = mixdown(stems, keys)
            x = torch.from_numpy(mix).unsqueeze(0).to(dev)
            feat, p = tf.features(x)
            dbm = 10 * torch.log10(p["M"] + 1e-12)
            mx = dbm.max(dim=2, keepdim=True).values
            active = (dbm > torch.maximum(torch.full_like(mx, -80.0), mx - 60.0)).float()
            dom, pitch, sal = targets(stems, tf, keys)
            pm = (pitch.amax(dim=2) > 0.5).float()
            out = net(feat)
            r = eval_batch(out, active, dom, pitch, out["pitch"], out["act"], pm)
            ap, npos, _ = pitch_ap(out["pitch"], pitch, pm[..., None].expand_as(out["pitch"]))
            rows, best = pitch_f1_sweep(out["pitch"][0].cpu(), pitch[0].cpu(),
                                        pm[0].cpu(), np.arange(0.05, 0.91, 0.05))
            r.update({"pitch_ap": ap, "pitch_pos": npos, "pitch_f1_best": best["f1"],
                      "pitch_thr_best": best["threshold"]})
            for k, v in r.items():
                acc.setdefault(kind, {}).setdefault(k, []).append(v)
            details.append({"dataset": kind, "track": name, "start": start,
                            "sources": len(keys), **{k: round(v, 4) for k, v in r.items()
                                                     if isinstance(v, (int, float))}})
    res = {k: {m: float(np.mean(v)) for m, v in d.items()} for k, d in acc.items()}
    return res, details


def build_clips(n: int, seed: int) -> list[tuple]:
    out = []
    mtracks = tracks(index_rows("test", "musdb"), "mix")
    for i, (name, _) in enumerate(sorted(mtracks.items())[: n]):
        start = 30.0 + 20.0 * (i % 3)
        out.append(("musdb", name, musdb_clip(name, start), start))
    sdirs = slakh_track_dirs("test")
    names = sorted(sdirs)[:n]
    for i, name in enumerate(names):
        stems = slakh_clip(sdirs[name], 30.0 + 20.0 * (i % 3))
        out.append(("slakh", name, stems, 30.0 + 20.0 * (i % 3)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--onnx", default="", help="exported model.onnx (v1 baseline)")
    ap.add_argument("--clips", type=int, default=6)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if not a.ckpt and not a.onnx:
        ap.error("pass --ckpt or --onnx")
    dev = torch.device(a.device)
    if a.onnx:
        net = OnnxFrameModel(a.onnx)
    else:
        ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        cfg = ck.get("config", {})
        hid = int(cfg.get("hid", 384))
        if cfg.get("arch", "v2") == "v3":
            from .model_v3 import AvisNetV3
            net = AvisNetV3(hid=hid)
        else:
            net = AvisNet(hid=hid)
        net.load_state_dict(ck["model"])
        net.eval().to(dev)
    tf = TorchFrontend(REPO / "shared" / "frontend" / "v1" / "weights.bin", dev)
    clips = build_clips(a.clips, seed=7)
    res, details = run_clips(net, tf, clips, dev)
    payload = {"ckpt": str(a.ckpt or a.onnx), "clips": a.clips, "by_dataset": res, "details": details}
    print(json.dumps(payload, indent=1))
    if a.out:
        p = Path(a.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
