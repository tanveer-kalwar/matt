#!/usr/bin/env python3
"""
MAT-Diff sampling script.

Usage:
    python -m mat_diff.sample --dataset wine --checkpoint results/matdiff_checkpoints/wine.pt
"""
import sys
import os
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mat_diff.manifold_diffusion import MATDiffPipeline
from mat_diff.config import get_matdiff_config


def main():
    parser = argparse.ArgumentParser(description="MAT-Diff Sampling")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Total samples per minority class (auto if None)")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    cfg = get_matdiff_config(args.dataset)

    pipeline = MATDiffPipeline(device=device)
    pipeline.load(args.checkpoint)

    # Load data to get class distribution
    from mat_diff.train import load_dataset_auto
    X_tr, y_tr, X_te, y_te = load_dataset_auto(args.dataset)
    pipeline.X_train = X_tr
    pipeline.y_train = y_tr

    if pipeline.privacy is not None:
        pipeline.privacy.fit_thresholds(X_tr, y_tr)

    if args.n_samples is not None:
        from collections import Counter
        cc = Counter(y_tr)
        maj = max(cc.values())
        n_per_class = {c: args.n_samples for c, v in cc.items() if v < maj}
    else:
        n_per_class = None

    print(f"\nSampling from {args.checkpoint}...")
    X_syn, y_syn = pipeline.sample(n_per_class=n_per_class)
    print(f"Generated: {X_syn.shape}")

    out_path = args.output or f"results/synthetic/{args.dataset}_matdiff.npz"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, X=X_syn, y=y_syn)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
