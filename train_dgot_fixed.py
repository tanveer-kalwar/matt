#!/usr/bin/env python3
"""
DGOT Fixed - Uses EXACT file structure and arguments DGOT expects
"""
import os
import sys
import subprocess
import numpy as np
import argparse


def setup_dgot_repo():
    """Clone and setup DGOT repository"""
    repo_dir = "/content/DGOT"
    
    if not os.path.isdir(repo_dir):
        print("  Cloning DGOT...")
        result = subprocess.run(
            "git clone https://github.com/MLneuers/DGOT.git /content/DGOT",
            shell=True, capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"  Clone failed: {result.stderr}")
            return None
    
    print("  Installing dependencies...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "torch", "scikit-learn", "xgboost", "pandas", "numpy", "pyyaml", "tqdm", "einops"],
        capture_output=True
    )
    
    return repo_dir


def prepare_data_for_dgot(dataset_name, X_train, y_train, repo_dir, fold=0):
    """
    DGOT expects data in: ./datasets/{dataset}/DGOT/exp{fold}/
    With shape (N, 1, features) for xtrain.npy
    Also needs TEST data
    """
    # Create DGOT directory
    data_dir = f"{repo_dir}/datasets/{dataset_name}/DGOT/exp{fold}"
    os.makedirs(data_dir, exist_ok=True)
    
    # CRITICAL: DGOT expects shape (N, 1, features), not (N, features)
    X_train_reshaped = X_train[:, None, :].astype(np.float32)
    
    # Normalize to [-1, 1] range
    X_min = X_train.min(axis=0, keepdims=True)
    X_max = X_train.max(axis=0, keepdims=True)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1.0
    X_train_norm = ((X_train - X_min) / X_range) * 2 - 1
    X_train_reshaped = X_train_norm[:, None, :].astype(np.float32)
    
    # Save training data
    np.save(f"{data_dir}/xtrain.npy", X_train_reshaped)
    np.save(f"{data_dir}/ytrain.npy", y_train.astype(np.int64))
    
    # Create TEST directory (required by DGOT)
    test_dir = f"{repo_dir}/datasets/{dataset_name}/TEST/exp{fold}"
    os.makedirs(test_dir, exist_ok=True)
    
    # Use a small portion as test data
    test_size = min(100, len(X_train) // 10)
    X_test_norm = X_train_norm[:test_size].astype(np.float32)
    y_test = y_train[:test_size].astype(np.int64)
    
    np.save(f"{test_dir}/xtest.npy", X_test_norm)
    np.save(f"{test_dir}/ytest.npy", y_test)
    
    print(f"      Saved: {data_dir}/xtrain.npy")
    print(f"      Shape: {X_train_reshaped.shape}")
    
    return X_train.shape[1], len(np.unique(y_train))


def train_dgot(dataset_name, n_features, n_classes, fold, repo_dir, device):
    """Run DGOT training with CORRECT arguments from train.py"""
    
    train_script = f"{repo_dir}/train.py"
    
    if not os.path.exists(train_script):
        print(f"      train.py not found at {train_script}")
        return False
    
    print(f"      Training DGOT...")
    
    # EXACT arguments from train.py argparse (lines 356-413)
    cmd = [
        sys.executable, train_script,
        "--dataset", dataset_name,
        "--exp", f"exp{fold}",
        "--class_num", str(n_classes),
        "--feature_len", str(n_features),
        "--num_epoch", "200",  # Reduced for faster training
        "--batch_size", "128",
        "--device", device,
        "--num_timesteps", "4",
        "--save_ckpt_every", "50",
        "--lr_d", "2e-3",
        "--lr_g", "5e-3",
    ]
    
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=1800, cwd=repo_dir
        )
        
        if result.returncode != 0:
            print(f"      Training failed with code {result.returncode}")
            if result.stderr:
                print(f"      STDERR (last 500 chars): {result.stderr[-500:]}")
            return False
        
        print(f"      Training complete")
        return True
        
    except subprocess.TimeoutExpired:
        print(f"      Training timeout")
        return False
    except Exception as e:
        print(f"      Training error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--X_train', type=str, required=True)
    parser.add_argument('--y_train', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    
    print(f"\nDGOT: {args.dataset}")
    print(f"{'='*60}")
    
    # Setup
    repo_dir = setup_dgot_repo()
    if repo_dir is None:
        sys.exit(1)
    
    # Load
    X_train = np.load(args.X_train)
    y_train = np.load(args.y_train)
    print(f"  Data: {X_train.shape}, Classes: {np.unique(y_train)}")
    
    # Prepare
    print(f"\n  Preparing data...")
    n_features, n_classes = prepare_data_for_dgot(
        args.dataset, X_train, y_train, repo_dir, fold=0
    )
    
    # Train
    print(f"\n  Training...")
    success = train_dgot(
        args.dataset, n_features, n_classes, fold=0, 
        repo_dir=repo_dir, device=args.device
    )
    
    if not success:
        print(f"\n  ✗ DGOT Training FAILED")
        print(f"  ✗ Check errors above for details")
        print(f"{'='*60}\n")
        sys.exit(1)
    
    # Training succeeded, now sample
    print(f"\n  Generating samples from trained model...")
    
    synthetic = None  # INITIALIZE HERE
    model_dir = f"{repo_dir}/saved_log/DGOT/{args.dataset}/exp0"
    
    # Check if netG.pth exists (trained model)
    if os.path.exists(f"{model_dir}/netG.pth"):
        print(f"      Model found, attempting manual sampling...")
        
        # Import DGOT modules and sample directly
        try:
            import torch
            sys.path.insert(0, repo_dir)
            from models.Generator import Unet
            from models.GaussionDiffusion import Posterior_Coefficients, sample_posterior
            import yaml
            
            # Load config
            with open(f"{model_dir}/configs.yaml") as f:
                dgot_args = argparse.Namespace(**yaml.load(f, Loader=yaml.FullLoader))
            
            device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
            
            # Load model
            attrvalues = np.eye(n_classes)
            netG = Unet(
                in_ch=1, out_ch=1,
                init_dim=n_features,
                nz=dgot_args.nz + attrvalues.shape[0],
                init_ch=dgot_args.init_ch,
                ch_mult=dgot_args.ch_mult,
                resnet_block_groups=dgot_args.rbg,
            ).to(device)
            
            netG.load_state_dict(torch.load(f"{model_dir}/netG.pth", map_location=device))
            netG.eval()
            
            pos_coeff = Posterior_Coefficients(dgot_args, device)
            
            # Calculate needed samples
            from collections import Counter
            cc = Counter(y_train)
            minority_label = min(cc, key=cc.get)
            needed = max(cc.values()) - cc[minority_label]
            
            # Generate samples for minority class
            with torch.no_grad():
                classnum = torch.tensor(attrvalues[[minority_label] * needed]).to(device)
                x_t_1 = torch.randn([needed, 1, n_features], device=device).float()
                
                x = x_t_1
                for i in reversed(range(dgot_args.num_timesteps)):
                    t = torch.full((x.size(0),), i, dtype=torch.int64).to(device)
                    latent_z = torch.randn(x.size(0), dgot_args.nz, device=device)
                    latent_zc = torch.cat([latent_z, classnum], 1).float()
                    
                    x_0 = netG(x, t, latent_zc)
                    x = sample_posterior(pos_coeff, x_0, x, t)
                
                synthetic = x.cpu().numpy().squeeze()
                
                # Denormalize
                X_min = X_train.min(axis=0, keepdims=True)
                X_max = X_train.max(axis=0, keepdims=True)
                X_range = X_max - X_min
                X_range[X_range == 0] = 1.0
                synthetic = (synthetic + 1) / 2 * X_range + X_min
                
                print(f"      Generated {len(synthetic)} samples from DGOT model")
                
        except Exception as e:
            print(f"      Sampling failed: {e}")
            import traceback
            traceback.print_exc()
            synthetic = None
    
    # Fail if sampling didn't work
    if synthetic is None:
        print(f"\n  ✗ DGOT Sampling FAILED")
        print(f"  ✗ Model trained but could not generate samples")
        print(f"{'='*60}\n")
        sys.exit(1)
    
    # Save
    output_dir = f"results_dgot/{args.dataset}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/synthetic_samples.npy"
    
    np.save(output_path, synthetic)
    print(f"\n  ✓ Saved {len(synthetic)} samples")
    print(f"  ✓ Location: {output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFATAL: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
