# R²M (Real-aware Residual Model Merging) – Training-free Kit

This is a minimal, training-free pipeline to merge specialist detectors into a **single** model:
1) **R²M merge** for the **encoder (backbone)**  
2) **Head**: averaged specialist head (2-logit)  
3) **Evaluation** via your existing test harness


---

## Prerequisites

- Your repo structure with:
  - `detectors/task_vectors.py` exposing `ImageEncoder`, `ClassificationHead`, `ImageClassifier`, `TaskVector`
  - `training/test.py` and dataset code as you posted
- Specialist checkpoints at `models`, each including `backbone.*` and `head.*` tensors.

## Quickstart

### 0) Edit config
Update `r2m/configs/df40_core_example.yaml`:
- `weight_root`: folder with `fs_k8.pth`, `fr_k8.pth`, `efs_k8.pth`
- `save`: output directory for merged models
- `tasks`: list of specialist stems

Key hyperparameters:
- `initial_rank_ratio` (r_fake): e.g., `0.05, 0.1, 0.2`
- `eta_norm_alpha` (α): e.g., `0.4, 0.6, 0.8` (η = α * ||τ_core||)
- `real_rank_ratio` (k_real): usually `1`

### 1) Merge specialists (R²M encoder)
```bash
python r2m/scripts/merge_r2m.py --cfg r2m/configs/df40_core_example.yaml
# => saves .../task_3__prior_0_5__r_0_1__alpha_0_6__CORE_backbone.pth
```

### 2) Build final classifier with averaged head
```bash
python r2m/scripts/merge_head_avg.py \
  --encoder ./training/df40_weights/merged/task_3__prior_0_5__r_0_1__alpha_0_6__CORE_backbone.pth \
  --heads ./training/df40_weights/tasks/3/fs_k8.pth \
          ./training/df40_weights/tasks/3/fr_k8.pth \
          ./training/df40_weights/tasks/3/efs_k8.pth \
  --out   ./training/df40_weights/final_model/merged_headC.pth
```

### 3) Evaluate with your existing tester
```bash
python r2m/scripts/eval_merged.py \
  --detector_yaml ./training/config/detector/clip.yaml \
  --weights ./training/df40_weights/final_model/merged_headC.pth \
  --datasets Celeb-DF-v2 FaceShifter DeeperForensics-1.0 \
  --save-json ./results/merged.json --save-csv ./results/merged.csv
```
