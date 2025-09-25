# -*- coding: utf-8 -*-
"""
Thin wrapper: calls your existing eval script with args.
Usage:
python r2m/scripts/eval_merged.py \
  --detector_yaml ./training/config/detector/clip.yaml \
  --weights ./training/df40_weights/final_model/merged_headC.pth \
  --datasets Celeb-DF-v2 FaceShifter DeeperForensics-1.0 \
  --save-json ./results/merged.json --save-csv ./results/merged.csv
"""
import argparse, subprocess, shlex, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector_yaml", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--save-json", default=None)
    ap.add_argument("--save-csv", default=None)
    args = ap.parse_args()

    cmd = [
        "python", "training/test.py",
        "--detector_path", args.detector_yaml,
        "--test_dataset", *args.datasets,
        "--weights_path", args.weights,
    ]
    if args.save_json:
        cmd += ["--save-json", args.save_json]
        os.makedirs(os.path.dirname(args.save_json), exist_ok=True)
    if args.save_csv:
        cmd += ["--save-csv", args.save_csv]
        os.makedirs(os.path.dirname(args.save_csv), exist_ok=True)

    print("[RUN]", " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()

