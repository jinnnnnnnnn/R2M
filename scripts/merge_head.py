# -*- coding: utf-8 -*-
"""
Head-(c): average 2-logit specialist heads, attach to a merged encoder -> final classifier .pth
Usage:
python r2m/scripts/merge_head_avg.py \
  --encoder ./training/df40_weights/merged/task_3__prior_0_5__r_0_1__alpha_0_6__CORE_backbone.pth \
  --heads ./training/df40_weights/tasks/3/fs_k8.pth ./training/df40_weights/tasks/3/fr_k8.pth ./training/df40_weights/tasks/3/efs_k8.pth \
  --out  ./training/df40_weights/final_model/merged_headC.pth
"""
import os, argparse, torch
from typing import Sequence, Tuple
from detectors.task_vectors import ImageEncoder, ClassificationHead

def _extract_Wb_from_specialist_headpath(p: str) -> Tuple[torch.Tensor, torch.Tensor]:
    ckpt = torch.load(p, map_location="cpu")
    W = ckpt["head.weight"] if "head.weight" in ckpt else ckpt["head.weight".replace("head.","head.")]
    b = ckpt.get("head.bias", None)
    if b is None:
        b = torch.zeros(W.shape[0], dtype=W.dtype)
    return W.cpu(), b.cpu()

def _avg_heads(paths: Sequence[str]) -> ClassificationHead:
    W_list, b_list = [], []
    out_dim, in_dim = None, None
    for p in paths:
        W, b = _extract_Wb_from_specialist_headpath(p)
        if out_dim is None:
            out_dim, in_dim = W.shape
            assert out_dim == 2, f"expected 2-logit heads, got {out_dim}"
        else:
            assert W.shape == (out_dim, in_dim), f"shape mismatch: {W.shape} vs {(out_dim, in_dim)}"
        W_list.append(W); b_list.append(b)
    Wm = torch.stack(W_list, 0).mean(0).contiguous()
    bm = torch.stack(b_list, 0).mean(0).contiguous()
    return ClassificationHead(normalize=False, weights=Wm, biases=bm)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--heads", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    enc = ImageEncoder.load(args.encoder).cpu().eval()
    head = _avg_heads(args.heads)
    clf = enc.__class__.ImageClassifier if hasattr(enc.__class__, "ImageClassifier") else None
    if clf is None:
        # detectors.task_vectors.ImageClassifier is available in your code
        from detectors.task_vectors import ImageClassifier
        clf = ImageClassifier
    model = clf(enc, head)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model.save(args.out)
    print(f"[Head-C] saved final classifier: {args.out}")

if __name__ == "__main__":
    main()

