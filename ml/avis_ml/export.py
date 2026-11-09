"""Export M-v1 to ONNX + model.json + C# golden outputs."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .frontend import TorchFrontend
from .mixer import MixConfig, ProceduralMixer, make_labels
from .model import AvisNet, StepWrapper

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=REPO).decode().strip()
    except Exception:
        return "unknown"


def compute_pca(net, tf, mixer, device, frames=1200):
    net.eval()
    embs = []
    with torch.no_grad():
        for i in range(4):
            b = mixer.batch(4, seed=900 + i)
            feat, active, pitch, sal, dom = make_labels(b["mix"], b["dom"], b["pitch"], b["sal"], tf)
            out = net(feat)
            e = out["emb"][active > 0.5].float().cpu().numpy()
            embs.append(e)
    X = np.concatenate(embs, axis=0)
    mean = X.mean(axis=0)
    Xc = X - mean
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    comps = vt[:2]
    return {"mean": mean.tolist(), "components": comps.tolist(), "n": int(X.shape[0])}


def export(ckpt_path, out_dir, device="cuda", skip_parity=False):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {})
    net = AvisNet()
    net.load_state_dict(ck["model"])
    net.eval().to(device)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    wrapper = StepWrapper(net).eval()
    feat = torch.randn(1, 3, 192, device=device)
    h = torch.zeros(2, 1, 384, device=device)
    onnx_path = out / "model.onnx"
    torch.onnx.export(
        wrapper, (feat, h), str(onnx_path),
        input_names=["feat", "h_in"],
        output_names=["emb", "act", "pitch", "sal", "h_out"],
        opset_version=17, dynamo=False, do_constant_folding=True,
    )
    print("onnx exported:", onnx_path.stat().st_size, "bytes", flush=True)

    # parity on 300 sequential frames
    parity = {"frames": 300}
    if not skip_parity:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        sess = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
        rng = np.random.default_rng(5)
        xs = torch.from_numpy(rng.standard_normal((300, 3, 192)).astype(np.float32)).to(device)
        h_t = torch.zeros(2, 1, 384, device=device)
        h_o = np.zeros((2, 1, 384), dtype=np.float32)
        diffs = []
        t0 = time.time()
        with torch.no_grad():
            for i in range(300):
                f = xs[i:i + 1]
                emb_t, act_t, pit_t, sal_t, h_t = net.step(f, h_t)
                emb_o, act_o, pit_o, sal_o, h_o = sess.run(None, {
                    "feat": f.cpu().numpy(), "h_in": h_o})
                diffs.append(max(float((emb_t.cpu().numpy() - emb_o).max()),
                                 float((act_t.cpu().numpy() - act_o).max()),
                                 float((pit_t.cpu().numpy() - pit_o).max()),
                                 float((sal_t.cpu().numpy() - sal_o).max())))
        parity.update({"max_abs": max(diffs), "mean_abs": float(np.mean(diffs)),
                       "ms_per_frame": (time.time() - t0) / 300 * 1000})
        print("onnx parity:", parity, flush=True)

    # save C# golden model io (first 300 random frames, deterministic)
    rng = np.random.default_rng(77)
    gx = rng.standard_normal((300, 3, 192)).astype(np.float32) * 0.5
    h_t = torch.zeros(2, 1, 384, device=device)
    embs, acts, pits, sals = [], [], [], []
    with torch.no_grad():
        for i in range(300):
            e, a, p, s, h_t = net.step(torch.from_numpy(gx[i:i + 1]).to(device), h_t)
            embs.append(e.cpu().numpy()); acts.append(a.cpu().numpy())
            pits.append(p.cpu().numpy()); sals.append(s.cpu().numpy())
    g = REPO / "shared" / "golden" / "model"
    g.mkdir(parents=True, exist_ok=True)
    gx.tofile(g / "input.f32")
    np.concatenate(embs).astype("<f4").tofile(g / "expected_emb.f32")
    np.concatenate(acts).astype("<f4").tofile(g / "expected_act.f32")
    np.concatenate(pits).astype("<f4").tofile(g / "expected_pitch.f32")
    np.concatenate(sals).astype("<f4").tofile(g / "expected_sal.f32")
    (g / "meta.json").write_text(json.dumps({"frames": 300, "bands": 192, "emb": 20,
                                             "h_layers": 2, "h_size": 384}, indent=1), encoding="utf-8")

    tf = TorchFrontend(REPO / "shared" / "frontend" / "v1" / "weights.bin", device=device)
    mixer = ProceduralMixer(MixConfig(seconds=4.0, max_sources=8), device=device)
    pca = compute_pca(net, tf, mixer, device)
    model_json = {
        "version": out.name,
        "created": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "frontend_version": "FE-v1",
        "inputs": [
            {"name": "feat", "shape": [1, 3, 192], "dtype": "float32"},
            {"name": "h_in", "shape": [2, 1, 384], "dtype": "float32"},
        ],
        "outputs": [
            {"name": "emb", "shape": [1, 192, 20], "dtype": "float32"},
            {"name": "act", "shape": [1, 192], "dtype": "float32"},
            {"name": "pitch", "shape": [1, 192], "dtype": "float32"},
            {"name": "sal", "shape": [1, 192], "dtype": "float32"},
            {"name": "h_out", "shape": [2, 1, 384], "dtype": "float32"},
        ],
        "color_pca": pca,
        "metrics": {"onnx_parity": parity, "train_step": ck.get("step"), "best_val": ck.get("best")},
        "train_config": cfg,
    }
    (out / "model.json").write_text(json.dumps(model_json, indent=1), encoding="utf-8")
    import shutil
    shutil.copy(REPO / "shared" / "frontend" / "v1" / "weights.bin", out / "weights.bin")
    shutil.copy(REPO / "shared" / "frontend" / "v1" / "frontend.json", out / "frontend.json")
    print("exported model.json, weights.bin, frontend.json ->", out, flush=True)
    return model_json


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-parity", action="store_true")
    a = ap.parse_args()
    export(a.ckpt, a.out, a.device, a.skip_parity)
