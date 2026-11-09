"""M-v1 training entry point. Causal model, GPU procedural mixer."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import torch
import yaml

from .frontend import TorchFrontend
from .losses import compute_losses
from .mixer import MixConfig, ProceduralMixer, make_labels
from .model import AvisNet, count_params

ROOT = Path(__file__).resolve().parents[1]


def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build(cfg, device):
    tf = TorchFrontend(ROOT.parent / "shared" / "frontend" / "v1" / "weights.bin", device=device)
    mixer = ProceduralMixer(MixConfig(seconds=float(cfg.get("seconds", 4.0)),
                                      max_sources=int(cfg.get("max_sources", 8))), device=device)
    net = AvisNet().to(device)
    return tf, mixer, net


def lr_at(step, base, warmup, total):
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * base * (1 + math.cos(math.pi * p))


def save_ckpt(path, net, opt, step, cfg, best, scaler=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": step, "model": net.state_dict(), "opt": opt.state_dict(),
        "config": cfg, "best": best,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }, path)


def load_ckpt(path, net, opt=None):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net.load_state_dict(ck["model"])
    if opt is not None and "opt" in ck:
        opt.load_state_dict(ck["opt"])
    return ck


def evaluate(net, tf, mixer, cfg, device, seed=12345):
    net.eval()
    with torch.no_grad():
        b = mixer.batch(int(cfg.get("val_batch", 4)), seed=seed)
        feat, active, pitch, sal, dom = make_labels(b["mix"], b["dom"], b["pitch"], b["sal"], tf)
        out = net(feat)
        ls = compute_losses(out, active, pitch, sal, dom)
    net.train()
    return {k: float(v) for k, v in ls.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "smoke.yaml"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    device = torch.device(args.device)
    out_dir = ROOT / "models" / (args.tag or cfg.get("tag", "smoke"))
    out_dir.mkdir(parents=True, exist_ok=True)

    tf, mixer, net = build(cfg, device)
    base_lr = float(cfg.get("lr", 1e-3))
    total = args.steps or int(cfg.get("steps", 500))
    bs = int(cfg.get("batch_size", 8))
    opt = torch.optim.AdamW(net.parameters(), lr=base_lr, weight_decay=float(cfg.get("wd", 1e-4)))
    start = 0
    best = 1e9
    ck_last = out_dir / "ckpt_last.pt"
    if args.resume and ck_last.exists():
        ck = load_ckpt(ck_last, net, opt)
        start = int(ck["step"])
        best = float(ck.get("best", 1e9))
        print(f"[resume] step={start} best={best:.4f}", flush=True)
    scaler = torch.cuda.amp.GradScaler(enabled=(bool(cfg.get("amp", True)) and device.type == "cuda"))
    use_amp = scaler.is_enabled()
    csv_path = out_dir / "train.csv"
    new_csv = not csv_path.exists() or start == 0
    fcsv = open(csv_path, "w", newline="", encoding="utf-8") if new_csv else open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(fcsv)
    if new_csv:
        writer.writerow(["step", "loss", "l_dc", "l_act", "l_pitch", "l_sal", "lr", "sec_per_step"])

    print(f"params={count_params(net)} device={device} steps={total} bs={bs}", flush=True)
    t0 = time.time()
    last = t0
    for step in range(start, total):
        lr = lr_at(step, base_lr, int(cfg.get("warmup", 1000)), total)
        for g in opt.param_groups:
            g["lr"] = lr
        b = mixer.batch(bs)
        feat, active, pitch, sal, dom = make_labels(b["mix"], b["dom"], b["pitch"], b["sal"], tf)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp, dtype=torch.float16):
            out = net(feat)
        with torch.autocast(device_type=device.type, enabled=False):
            ls = compute_losses({k: (v.float() if torch.is_tensor(v) else v) for k, v in out.items()},
                                active.float(), pitch.float(), sal.float(), dom)
            loss = ls["loss"]
        if not torch.isfinite(loss):
            print(f"[warn] non-finite loss at {step}; disabling AMP", flush=True)
            use_amp = False
            scaler = torch.cuda.amp.GradScaler(enabled=False)
            opt.zero_grad(set_to_none=True)
            out = net(feat)
            with torch.autocast(device_type=device.type, enabled=False):
                ls = compute_losses({k: (v.float() if torch.is_tensor(v) else v) for k, v in out.items()},
                                    active.float(), pitch.float(), sal.float(), dom)
                loss = ls["loss"]
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
        now = time.time()
        sps = now - last
        last = now
        if step % int(cfg.get("log_every", 20)) == 0:
            writer.writerow([step, f"{float(loss):.5f}", f"{float(ls['l_dc']):.5f}",
                             f"{float(ls['l_act']):.5f}", f"{float(ls['l_pitch']):.5f}",
                             f"{float(ls['l_sal']):.5f}", f"{lr:.6f}", f"{sps:.3f}"])
            fcsv.flush()
            print(f"step {step} loss {float(loss):.4f} dc {float(ls['l_dc']):.4f} "
                  f"act {float(ls['l_act']):.4f} pitch {float(ls['l_pitch']):.4f} "
                  f"sal {float(ls['l_sal']):.4f} lr {lr:.2e} {sps:.2f}s", flush=True)
        if (step + 1) % int(cfg.get("ckpt_every", 2000)) == 0 or step + 1 == total:
            val = evaluate(net, tf, mixer, cfg, device)
            save_ckpt(ck_last, net, opt, step + 1, cfg, min(best, val["loss"]), scaler)
            if val["loss"] < best:
                best = val["loss"]
                save_ckpt(out_dir / "ckpt_best.pt", net, opt, step + 1, cfg, best, scaler)
            (out_dir / "val.json").write_text(json.dumps(val, indent=1), encoding="utf-8")
            print(f"[ckpt] step {step+1} val {val['loss']:.4f}", flush=True)
        if step % 100 == 0 and step > start:
            elapsed = time.time() - t0
            rate = (step - start) / max(elapsed, 1e-6)
            print(f"[eta] {rate:.2f} steps/s, elapsed {elapsed/60:.1f} min", flush=True)
    fcsv.close()
    print(f"done in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
