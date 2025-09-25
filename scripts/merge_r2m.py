# -*- coding: utf-8 -*-
"""
R²M: training-free merge of specialists into a single encoder (backbone only).
- Loads zero-shot encoder via detectors.task_vectors.ImageEncoder
- Builds task vectors from specialist .pths
- Real-aware decomposition to get τ_core and truncated residual aggregation
- Saves merged encoder as *.pth (ImageEncoder.save)

Usage:
python r2m/scripts/merge_r2m.py --cfg r2m/configs/df40_core_example.yaml
"""
import os, gc, math, argparse
from collections import OrderedDict
from typing import List, Tuple, Dict
import torch
import numpy as np
from omegaconf import OmegaConf

# === project imports you already have ===
from detectors.task_vectors import ImageEncoder, TaskVector, ClassificationHead  # noqa

CPU = "cpu"
EPS = 1e-8

def _dict_flatten(d: OrderedDict) -> torch.Tensor:
    return torch.cat([p.reshape(-1).float() for p in d.values()])

def _dict_l2(d: OrderedDict) -> torch.Tensor:
    v = _dict_flatten(d)
    return torch.linalg.vector_norm(v) + 1e-12

def _dict_scale(d: OrderedDict, s: float) -> OrderedDict:
    return OrderedDict((k, v * s) for k, v in d.items())

def _dict_mean(dicts: List[OrderedDict]) -> OrderedDict:
    keys = dicts[0].keys()
    out = OrderedDict()
    for k in keys:
        out[k] = sum(d[k] for d in dicts) / len(dicts)
    return out

@torch.no_grad()
def decompose_real_fake(task_vecs: List[OrderedDict],
                        rank: int = 1,
                        device: str = "cuda",
                        include_all_layers: bool = True,
                        exclude_patterns: Tuple[str,...] = ("ln","bias"),
                        block_norm: str = "fro_sqrtP",
                        real_blend_beta: float = 1.0):
    """
    Same behavior as your long variant:
      - Estimate global 'Real' subspace over included layers
      - tau_core = P_real * mean_tau   (with optional beta-blending with mean)
      - tau_i^real, tau_i^fake decomposition
    Returns: tau_core (OrderedDict), fakes (List[OrderedDict])
    """
    assert len(task_vecs) > 0
    N = len(task_vecs)
    k = min(rank, N)
    keys = list(task_vecs[0].keys())

    def use_layer(name: str) -> bool:
        if any(p in name for p in exclude_patterns):
            return False
        return True if include_all_layers else ("attn" in name or "mlp" in name)

    # Build C = sum_l (Â_l Â_l^T)
    C = torch.zeros((N, N), dtype=torch.float32, device=device)
    gammas = {}
    A_cache = {}
    for key in keys:
        if not use_layer(key): continue
        A_rows = [tv[key].to(device=device, dtype=torch.float32).flatten() for tv in task_vecs]
        A = torch.stack(A_rows, dim=0)  # N x P
        if block_norm == "none":
            gamma = 1.0
        elif block_norm == "fro":
            gamma = 1.0 / (torch.linalg.norm(A, ord="fro") + EPS)
        elif block_norm == "fro_sqrtP":
            P = A.shape[1]
            gamma = math.sqrt(P) / (torch.linalg.norm(A, ord="fro") + EPS)
        else:
            raise ValueError(f"unknown block_norm: {block_norm}")
        gammas[key] = float(gamma)
        A_cache[key] = A
        C += (A * gamma) @ A.t()

    if k == 0 or torch.all(C == 0):
        tau_core = OrderedDict((k, torch.zeros_like(task_vecs[0][k])) for k in keys)
        fakes = [OrderedDict((k, tv[k].clone()) for k in keys) for tv in task_vecs]
        return tau_core, fakes

    # eigendecomp
    evals, U = torch.linalg.eigh(C)
    idx = torch.argsort(evals, descending=True)
    lam_k = torch.clamp(evals[idx][:k], min=EPS)
    U_k = U[:, idx][:, :k]
    U_scaled = U_k / lam_k.sqrt().unsqueeze(0)

    # tau_core
    mean_tau = OrderedDict((k, torch.mean(torch.stack([tv[k] for tv in task_vecs], dim=0), dim=0)) for k in keys)
    y_core = torch.zeros((k,), dtype=torch.float32, device=device)
    for key in keys:
        if not use_layer(key): continue
        A = A_cache[key]
        A_norm = A * gammas[key]
        bar_tau_l = mean_tau[key].to(device, dtype=torch.float32).flatten()
        a = A_norm @ bar_tau_l
        y_core += U_scaled.t() @ a
    w_core = U_scaled @ y_core

    tau_core = OrderedDict()
    for key in keys:
        if use_layer(key):
            A = A_cache[key]
            core_flat = (A.t() @ w_core) * gammas[key]
            core_proj = core_flat.view(task_vecs[0][key].shape).to(task_vecs[0][key].device)
            if real_blend_beta != 1.0:
                tau_core[key] = (1.0 - real_blend_beta) * mean_tau[key] + real_blend_beta * core_proj
            else:
                tau_core[key] = core_proj
        else:
            tau_core[key] = mean_tau[key].clone()

    # real/fake split
    fakes = []
    for i in range(N):
        fake_i = OrderedDict()
        for key in keys:
            fake_i[key] = task_vecs[i][key] - tau_core[key].to(task_vecs[i][key].device)
        fakes.append(fake_i)
    return tau_core, fakes

def load_zero_shot_state(config, device):
    enc = ImageEncoder(args=config, keep_lang=False).to(device).eval()
    sd = enc.state_dict()
    sd = {k.replace("model.", ""): v for k, v in sd.items()}
    del enc; gc.collect()
    return sd

def load_task_vectors(config, device) -> Tuple[List[TaskVector], List[Dict[str,torch.Tensor]]]:
    zsd = load_zero_shot_state(config, device)
    tvs = []
    heads = []
    for t in config.tasks:
        p = os.path.join(config.weight_root, t + ".pth")
        ckpt = torch.load(p, map_location=device)
        bb = {k[len("backbone."):]: v for k, v in ckpt.items() if k.startswith("backbone.")}
        hd = {k[len("head."):]: v for k, v in ckpt.items() if k.startswith("head.")}
        tvs.append(TaskVector(config, zsd, bb, task=t).to(device))
        heads.append(hd)
    return tvs, heads

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    args = ap.parse_args()
    cfg = OmegaConf.load(args.cfg)
    device = "cuda" if (getattr(cfg, "device", "auto") == "auto" and torch.cuda.is_available()) else getattr(cfg, "device", "cpu")

    tvs, heads = load_task_vectors(cfg, device)
    tv_dicts = [tv.vector for tv in tvs]

    tau_core, fakes = decompose_real_fake(
        tv_dicts,
        rank=cfg.real_rank_ratio,
        device=device if torch.cuda.is_available() else "cpu",
        include_all_layers=True,
        exclude_patterns=("ln","bias"),
        block_norm="fro_sqrtP",
        real_blend_beta=float(getattr(cfg, "real_blend_beta", 1.0)),
    )

    # mean-rescale of truncated residuals, then uniform average
    norms = torch.stack([_dict_l2(d) for d in fakes])
    m_mean = norms.mean()
    rescaled = [_dict_scale(d, float((m_mean / (n + 1e-8)).item())) for d, n in zip(fakes, norms)]
    tau_res_merge = _dict_mean(rescaled)

    # η_eff = α * ||τ_core||
    alpha = float(getattr(cfg, "eta_norm_alpha", cfg.prior))
    eta = alpha * float(_dict_l2(tau_core).item())

    # θ* = θ0 + τ_core + η τ_res_merge
    zsd = load_zero_shot_state(cfg, device)
    merged = {}
    for k in zsd.keys():
        base = zsd[k]
        add_core = tau_core.get(k, None)
        add_res  = tau_res_merge.get(k, None)
        v = base.clone()
        if add_core is not None:
            v = v + add_core.to(v.device, v.dtype)
        if add_res is not None:
            v = v + add_res.to(v.device, v.dtype) * eta
        merged[k] = v

    out_dir = cfg.save
    os.makedirs(out_dir, exist_ok=True)
    name = f"task_{len(cfg.tasks)}__prior_{str(cfg.prior).replace('.','_')}__r_{str(cfg.initial_rank_ratio).replace('.','_')}__alpha_{str(alpha).replace('.','_')}__CORE_backbone.pth"

    enc = ImageEncoder.load_from_state_dict(cfg, merged)
    enc.save(os.path.join(out_dir, name))
    print(f"[R2M] saved: {os.path.join(out_dir, name)}")
    print(f"[R2M] ||tau_core||={float(_dict_l2(tau_core)):.3e}, eta={eta:.3e}, m_mean={float(m_mean):.3e}")

if __name__ == "__main__":
    main()

