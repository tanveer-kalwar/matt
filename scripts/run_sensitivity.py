#!/usr/bin/env python3
"""
Sensitivity analysis for TabDDPM-Aug hyperparameters.

Tests:
1. Overgeneration factor λ ∈ {1.0, 1.5, 1.8, 3.0}
2. Hardness threshold τ_hard ∈ {0.5, 0.6, 0.7, 0.8, 0.9}

Usage:
    python scripts/run_sensitivity.py --dataset credit --mode overgen
    python scripts/run_sensitivity.py --dataset credit --mode threshold
"""
import sys
from pathlib import Path
import argparse
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from TabDDPM_Aug.data_loader import load_dataset
from TabDDPM_Aug.config import get_config
from TabDDPM_Aug.experiments.sensitivity import (
    run_sensitivity_overgeneration,
    run_sensitivity_threshold
)


def main():
    parser = argparse.ArgumentParser(
        description='Run sensitivity analysis for TabDDPM-Aug hyperparameters'
    )
    parser.add_argument(
        '--dataset', type=str, default='credit',
        choices=['adult', 'pima', 'credit'],
        help='Dataset to use (default: credit, matching the paper and avoiding multiple-dataset sweeps)'
    )
    parser.add_argument(
        '--mode', type=str, required=True,
        choices=['overgen', 'threshold', 'both'],
        help='Sensitivity analysis mode'
    )
    args = parser.parse_args()
    
    print("="*80)
    print("TabDDPM-Aug: Sensitivity Analysis")
    print(f"Dataset: {args.dataset}")
    print(f"Mode: {args.mode}")
    print("="*80)
    
    # Force Credit dataset for efficiency (as in paper).
    # If you want to run sensitivity on another dataset, remove this block.
    if args.dataset != 'credit':
        print(f"\nWARNING: Forcing credit dataset for sensitivity analysis")
        print(f"(Requested: {args.dataset}, Using: credit)\n")
        args.dataset = 'credit'
    
    df = load_dataset(args.dataset)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    main_config = get_config(args.dataset)
    
    # Run requested analyses
    if args.mode == 'overgen':
        print("\n" + "="*80)
        print("EXPERIMENT 1: OVERGENERATION FACTOR (λ)")
        print("="*80)
        results_overgen = run_sensitivity_overgeneration(df, device, main_config)
        
    elif args.mode == 'threshold':
        print("\n" + "="*80)
        print("EXPERIMENT 2: HARDNESS THRESHOLD (τ_hard)")
        print("="*80)
        results_threshold = run_sensitivity_threshold(df, device, main_config)
        
    else:  # both
        print("\n" + "="*80)
        print("EXPERIMENT 1: OVERGENERATION FACTOR (λ)")
        print("="*80)
        results_overgen = run_sensitivity_overgeneration(df, device, main_config)
        
        print("\n" + "="*80)
        print("EXPERIMENT 2: HARDNESS THRESHOLD (τ_hard)")
        print("="*80)
        results_threshold = run_sensitivity_threshold(df, device, main_config)
    
    print("\n" + "="*100)
    print("SENSITIVITY ANALYSIS COMPLETE")
    print("="*100)

if __name__ == '__main__':
    main()
