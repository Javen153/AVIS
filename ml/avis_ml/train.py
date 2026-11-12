"""M-v2 training: from scratch on procedural + real stems, EMA + composite selection."""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch
import yaml

from .frontend import TorchFrontend
from .losses import UncertaintyWeights, compute_losses
from .metrics import embedding_cosine
from .mixer import MixConfig, ProceduralMixer
from .mixer_v2 import PROC_P, RealMixer
from .model import HID, AvisNet, count_params
from .validate import (composite, evaluate, head_grad_norms, labels_of,
                       make_val_sets)

ROOT = Path(__file__).resolve().parents[1]


def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(cfg):
    """Architecture switch; both variants share the ONNX step contract."""
    if cfg.get("arch", "v2") == "v3":
        from .model_v3 import HID as HID_V3
        from .model_v3 import AvisNetV3, count_params as count_v3
        net = AvisNetV3(hid=int(cfg.get("hid", HID_V3)))
        return net, int(cfg.get("hid", HID_V3)), count_v3
    net = AvisNet(hid=int(cfg.get("hid", HID)))
    return net, int(cfg.get("hid", HID)), count_params


def build(cfg, device):
    torch.manual_seed(int(cfg.get("seed", 0)))
    tf = TorchFrontend(ROOT.parent / "shared" / "frontend" / "v1" / "weights.bin", device=device)
    mixer = ProceduralMixer(MixConfig(seconds=float(cfg.get("seconds", 4.0)),
                                      max_sources=int(cfg.get("max_sources", 8))), device=device)
    net, hid, _ = build_model(cfg)
    return tf, mixer, net.to(device), hid


def lr_at(step, base, warmup, total):
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * base * (1 + math.cos(math.pi * p))


def save_ckpt(path, net, opt, step, cfg, best, scaler=None, ema=None, uw=None,
              model_state=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    state = model_state if model_state is not None else net.state_dict()
    torch.save({
        "step": step, "model": state, "opt": opt.state_dict(),
        "config": cfg, "best": best,
        "ema": ema if ema is not None else state,
        "uw": uw.state_dict() if uw is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }, path)


def load_ckpt(path, net, opt=None, uw=None):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net.load_state_dict(ck["model"])
    if opt is not None and "opt" in ck:
        opt.load_state_dict(ck["opt"])
    if uw is not None and ck.get("uw") is not None:
        uw.load_state_dict(ck["uw"])
    return ck


def make_real_mixer(cfg, tf, base, splits, workers):
    from .real_data import RealSourceSampler, load_pool
    from .data_index import data_root

    pool = load_pool(data_root() / "index.jsonl", splits)
    stereo = cfg.get("arch") == "v3" and bool(cfg.get("stereo", True))
    sampler = RealSourceSampler(pool, batch=int(cfg.get("real_batch", 16)), workers=workers,
                                seed=int(cfg.get("seed", 0)), stereo=stereo)
    if cfg.get("arch") == "v3":
        from .mixer_v3 import RealMixerV3
        return RealMixerV3(base, sampler, tf, proc_p=float(cfg.get("proc_share", PROC_P)),
                           distort_p=float(cfg.get("mix_distort_p", 0.5)),
                           hard_p=float(cfg.get("hard_p", 0.35)), stereo=stereo)
    return RealMixer(base, sampler, tf, proc_p=float(cfg.get("proc_share", PROC_P)),
                     distort_p=float(cfg.get("mix_distort_p", 0.5)))


def proc_share_at(step, total, cfg):
    stage_a = int(cfg.get("stage_a_steps", 0))
    if step < stage_a:
        return 1.0
    start = float(cfg.get("proc_share_start", 0.75))
    end = float(cfg.get("proc_share_end", 0.5))
    span = max(1, total - stage_a)
    frac = min(1.0, max(0.0, (step - stage_a) / span))
    return start + (end - start) * frac


def ema_state(net):
    return {k: v.detach().clone().float() for k, v in net.state_dict().items()}


def ema_update(ema, net, decay, step=None):
    # warm up the decay early so validation tracks training after a few hundred
    # steps; long runs still converge to the configured horizon (timm-style EMA).
    if step is not None:
        decay = min(decay, (1.0 + step) / (10.0 + step))
    with torch.no_grad():
        for k, v in net.state_dict().items():
            e = ema.get(k)
            if e is None:
                ema[k] = v.detach().clone().float()
            elif e.dtype.is_floating_point:
                e.mul_(decay).add_(v.detach().float(), alpha=1.0 - decay)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "v2.yaml"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    device = torch.device(args.device)
    out_dir = ROOT / "models" / (args.tag or cfg.get("tag", "v2"))
    out_dir.mkdir(parents=True, exist_ok=True)

    tf, mixer, net, hid = build(cfg, device)
    base = mixer
    if cfg.get("real"):
        mixer = make_real_mixer(cfg, tf, base, ("train",), workers=int(cfg.get("real_workers", 4)))
    val_sets = make_val_sets(cfg, tf, base)
    eval_net, _, _ = build_model(cfg)
    eval_net = eval_net.to(device)
    uw = None
    if cfg.get("uncertainty", True):
        uw = UncertaintyWeights(4, cfg.get("log_var_init")).to(device)
    params = list(net.parameters()) + (list(uw.parameters()) if uw is not None else [])
    base_lr = float(cfg.get("lr", 1e-3))
    total = args.steps or int(cfg.get("steps", 8000))
    bs = int(cfg.get("batch_size", 8))
    opt = torch.optim.AdamW(params, lr=base_lr, weight_decay=float(cfg.get("wd", 1e-4)))
    start = 0
    best = -1e9
    ck_last = out_dir / "ckpt_last.pt"
    ck_best = out_dir / "ckpt_best.pt"
    ema = None
    if args.resume and ck_last.exists():
        ck = load_ckpt(ck_last, net, opt, uw)
        start = int(ck["step"])
        ema = ck.get("ema")
        best = float(ck.get("best", -1e9))
        print(f"[resume] step={start} best={best:.4f}", flush=True)
    elif cfg.get("init_from"):
        src = Path(cfg["init_from"])
        if not src.is_absolute():
            src = ROOT / src
        ck = torch.load(src, map_location="cpu", weights_only=False)
        try:
            net.load_state_dict(ck["model"])
        except RuntimeError as e:
            raise SystemExit(f"init_from {src.name} is incompatible ({e}); train from scratch") from e
        print(f"[init] weights from {src.name}", flush=True)
    if ema is None:
        ema = ema_state(net)

    def sync_eval():
        eval_net.load_state_dict({k: v.to(device) for k, v in ema.items()})
        return eval_net

    scaler = torch.amp.GradScaler("cuda", enabled=(bool(cfg.get("amp", True))
                                                   and device.type == "cuda"))
    use_amp = scaler.is_enabled()
    clip = float(cfg.get("clip", 1.0))
    skip_loss = float(cfg.get("skip_loss", 25.0))
    lr_scale = 1.0
    bad_vals = 0
    bad_evals = 0
    consec_skip = 0
    recover_after = int(cfg.get("auto_recover", 0))
    patience = int(cfg.get("patience_val", 2))
    early_stop = int(cfg.get("early_stop", 3))
    val_noise = float(cfg.get("val_noise", 0.03))
    ema_decay = float(cfg.get("ema_decay", 0.999))
    margin_db = float(cfg.get("dc_margin_db", 3.0))

    def recover(tag: str) -> bool:
        nonlocal lr_scale, bad_vals, bad_evals, consec_skip
        if not ck_best.exists():
            return False
        load_ckpt(ck_best, net, opt, uw)
        for k, v in net.state_dict().items():
            ema[k] = v.detach().clone().float()
        lr_scale *= 0.5
        bad_vals = bad_evals = consec_skip = 0
        print(f"[recover] {tag}; rolled back to {ck_best.name}, lr_scale={lr_scale:.3f}",
              flush=True)
        return True

    # gradient norm at each private head output, see validate.head_grad_norms
    def _make_hook(name):
        def _hook(_m, _inp, out):
            out.retain_grad()
            net._head_grads[name] = out
        return _hook

    net._head_grads = {}
    for _name, _head in (("emb", net.head_emb), ("act", net.head_act),
                         ("pitch", net.head_pitch), ("sal", net.head_sal)):
        _head.register_forward_hook(_make_hook(_name))

    # initial baseline on the fixed set (composite, not loss); on resume keep the
    # stored best so a worse post-resume model cannot overwrite ckpt_best
    val = evaluate(sync_eval(), tf, val_sets, device, cfg, uw=uw, sweep=False,
                   cell_acc=bool(cfg.get("val_cell_acc", True)))
    init_score = composite(val)
    if init_score > best:
        best = init_score
        if start > 0:
            # resumed weights beat the stored best: refresh ckpt_best so the file
            # matches the in-memory best (D-037)
            save_ckpt(ck_best, net, opt, start, cfg, best, scaler, ema, uw,
                      model_state=ema)
    print(f"[init] composite {init_score:.4f} best {best:.4f} ap {val['pitch_ap']:.4f} "
          f"iou {val['act_iou']:.4f} dc {val['l_dc']:.4f}", flush=True)
    if start == 0:
        save_ckpt(ck_last, net, opt, 0, cfg, best, None, ema, uw)
        save_ckpt(ck_best, net, opt, 0, cfg, best, None, ema, uw, model_state=ema)
    (out_dir / "val.json").write_text(json.dumps(val, indent=1), encoding="utf-8")

    csv_path = out_dir / "train.csv"
    new_csv = not csv_path.exists() or start == 0
    fcsv = open(csv_path, "w" if new_csv else "a", newline="", encoding="utf-8")
    writer = csv.writer(fcsv)
    if new_csv:
        writer.writerow(["step", "loss", "l_dc", "l_act", "l_pitch", "l_sal", "lr",
                         "sec_per_step", "w_dc", "w_act", "w_pitch", "w_sal",
                         "g_emb", "g_act", "g_pitch", "g_sal", "pitch_frac", "pitch_max",
                         "emb_cos"])

    print(f"params={count_params(net)} hid={hid} device={device} steps={total} bs={bs} "
          f"real={bool(cfg.get('real'))}", flush=True)
    t0 = time.time()
    last = t0
    # Running mean of the solo-coherence term over batches that actually contain
    # a solo example (only ~32 % of batches do at hard_p 0.3 / solo 20 %), so the
    # log line reports the term's trend instead of a per-batch hit or miss.
    coh_sum = 0.0
    coh_n = 0
    for step in range(start, total):
        lr = lr_at(step, base_lr, int(cfg.get("warmup", 1000)), total) * lr_scale
        for g in opt.param_groups:
            g["lr"] = lr
        if isinstance(mixer, RealMixer):
            mixer.proc_p = proc_share_at(step, total, cfg)
        b = mixer.batch(bs)
        lab = labels_of(b, tf, margin_db)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp, dtype=torch.float16):
            out = net(lab["feat"])
        with torch.autocast(device_type=device.type, enabled=False):
            ls = compute_losses({k: (v.float() if torch.is_tensor(v) else v)
                                 for k, v in out.items()}, lab, cfg, uw)
            loss = ls["loss"]
        if not torch.isfinite(loss):
            print(f"[warn] non-finite loss at {step}; disabling AMP", flush=True)
            use_amp = False
            scaler = torch.amp.GradScaler("cuda", enabled=False)
            opt.zero_grad(set_to_none=True)
            out = net(lab["feat"])
            with torch.autocast(device_type=device.type, enabled=False):
                ls = compute_losses({k: (v.float() if torch.is_tensor(v) else v)
                                     for k, v in out.items()}, lab, cfg, uw)
                loss = ls["loss"]
        skipped = (not torch.isfinite(loss)) or float(loss.detach()) > skip_loss
        solo_n = 0.0 if lab.get("solo") is None else float(lab["solo"].sum())
        if not skipped and solo_n > 0.0:
            coh_sum += float(ls["l_coh"].detach())
            coh_n += 1
        gnorm = {"g_emb": 0.0, "g_act": 0.0, "g_pitch": 0.0, "g_sal": 0.0}
        if skipped:
            print(f"[warn] skip step {step}: loss {float(loss.detach()):.1f} > {skip_loss:.0f} "
                  f"| dc {float(ls['l_dc'].detach()):.2f} act {float(ls['l_act'].detach()):.2f} "
                  f"pitch {float(ls['l_pitch'].detach()):.2f} "
                  f"sal {float(ls['l_sal'].detach()):.2f}", flush=True)
            opt.zero_grad(set_to_none=True)
            consec_skip += 1
            if recover_after and consec_skip >= recover_after:
                recover(f"{consec_skip} consecutive skipped steps")
        else:
            consec_skip = 0
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                gnorm = head_grad_norms(net)
                finite = all(torch.isfinite(p.grad).all() for p in params
                             if p.grad is not None)
                if finite:
                    torch.nn.utils.clip_grad_norm_(params, clip)
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.zero_grad(set_to_none=True)
                    scaler.update()
            else:
                loss.backward()
                gnorm = head_grad_norms(net)
                finite = all(torch.isfinite(p.grad).all() for p in params
                             if p.grad is not None)
                if finite:
                    torch.nn.utils.clip_grad_norm_(params, clip)
                    opt.step()
                else:
                    opt.zero_grad(set_to_none=True)
            if finite:
                ema_update(ema, net, ema_decay, step)
            else:
                skipped = True
                consec_skip += 1
                print(f"[warn] non-finite gradient at {step}; skipped update "
                      f"| g {gnorm['g_emb']:.2f}/{gnorm['g_act']:.2f}/"
                      f"{gnorm['g_pitch']:.2f}/{gnorm['g_sal']:.2f}", flush=True)
                if recover_after and consec_skip >= recover_after:
                    recover(f"{consec_skip} consecutive non-finite gradients")
        now = time.time()
        sps = now - last
        last = now
        if not skipped and step % int(cfg.get("log_every", 20)) == 0:
            pstat = out["pitch"].float()
            frac = float((pstat > 0.5).float().mean())
            pmax = float(pstat.max().detach())
            cos = embedding_cosine(out["emb"], lab["active"], max_samples=1500)
            if uw is not None:
                w = [float(x) for x in uw.weights()]
            else:
                w = [float(x) for x in cfg.get("loss_weights", [1, .5, 1, .5])]
            lv = {k: float(ls[k].detach()) for k in
                  ("loss", "l_dc", "l_coh", "l_act", "l_pitch", "l_sal")}
            writer.writerow([step, f"{lv['loss']:.5f}", f"{lv['l_dc']:.5f}",
                             f"{lv['l_act']:.5f}", f"{lv['l_pitch']:.5f}",
                             f"{lv['l_sal']:.5f}", f"{lr:.6f}", f"{sps:.3f}",
                             f"{w[0]:.4f}", f"{w[1]:.4f}", f"{w[2]:.4f}", f"{w[3]:.4f}",
                             f"{gnorm['g_emb']:.3f}", f"{gnorm['g_act']:.3f}",
                             f"{gnorm['g_pitch']:.3f}", f"{gnorm['g_sal']:.3f}",
                             f"{frac:.4f}", f"{pmax:.4f}", f"{cos:.4f}"])
            fcsv.flush()
            print(f"step {step} loss {lv['loss']:.4f} dc {lv['l_dc']:.4f} "
                  f"coh {coh_sum / coh_n if coh_n else 0.0:.3f} "
                  f"act {lv['l_act']:.4f} pitch {lv['l_pitch']:.4f} "
                  f"sal {lv['l_sal']:.4f} lr {lr:.2e} {sps:.2f}s "
                  f"| g {gnorm['g_emb']:.2f}/{gnorm['g_act']:.2f}/{gnorm['g_pitch']:.2f}/"
                  f"{gnorm['g_sal']:.2f} p>0.5 {frac:.3f} max {pmax:.3f} cos {cos:.3f}",
                  flush=True)
        if (step + 1) % int(cfg.get("ckpt_every", 1000)) == 0 or step + 1 == total:
            val = evaluate(sync_eval(), tf, val_sets, device, cfg, uw=uw, sweep=False,
                           cell_acc=bool(cfg.get("val_cell_acc", True)))
            score = composite(val)
            improved = score > best + 1e-6
            final = step + 1 >= total
            if improved:
                best = score
                bad_vals = bad_evals = 0
            else:
                bad_evals += 1
                if best - score > 2.0 * val_noise:
                    bad_vals += 1
                    print(f"[warn] composite {score:.4f} is {best - score:.4f} below best "
                          f"(2 sigma = {2 * val_noise:.4f})", flush=True)
                else:
                    bad_vals = 0
            if not improved and not final and patience and bad_vals >= patience:
                recover(f"composite {score:.4f} below best {best:.4f}")
                val = evaluate(sync_eval(), tf, val_sets, device, cfg, uw=uw, sweep=False,
                               cell_acc=bool(cfg.get("val_cell_acc", True)))
                score = composite(val)
            save_ckpt(ck_last, net, opt, step + 1, cfg, best, scaler, ema, uw)
            if improved:
                save_ckpt(ck_best, net, opt, step + 1, cfg, best, scaler, ema, uw,
                          model_state=ema)
            (out_dir / "val.json").write_text(json.dumps(val, indent=1), encoding="utf-8")
            print(f"[ckpt] step {step+1} composite {score:.4f} best {best:.4f} "
                  f"ap {val['pitch_ap']:.4f} f1 {val['pitch_f1']:.4f} "
                  f"iou {val['act_iou']:.4f} dc {val['l_dc']:.4f} "
                  f"cell {val['cell_acc']:.4f} cos {val['emb_cos']:.3f} "
                  f"noise {val['batch_noise']['pitch_ap_std']:.4f} lr_scale={lr_scale:.3f}",
                  flush=True)
            if hasattr(mixer, "stats"):
                print(f"[mix] {mixer.stats}", flush=True)
            if not improved and not final and early_stop and bad_evals >= early_stop:
                print(f"[stop] {bad_evals} evaluations without improvement", flush=True)
                break
        if step % 100 == 0 and step > start:
            elapsed = time.time() - t0
            rate = (step - start) / max(elapsed, 1e-6)
            print(f"[eta] {rate:.2f} steps/s, elapsed {elapsed/60:.1f} min", flush=True)
    fcsv.close()
    print(f"done in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
