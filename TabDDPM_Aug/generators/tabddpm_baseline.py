"""
Original TabDDPM baseline (trained exclusively on minority samples).
"""
import numpy as np
import torch
import torch.nn as nn

try:
    from tab_ddpm.gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion
    from tab_ddpm.modules import MLPDiffusion
    TABDDPM_AVAILABLE = True
except ImportError:
    TABDDPM_AVAILABLE = False


def tabddpm_generator_original_baseline(X_minority_norm, n_needed, config, seed=42, device='cpu'):
    """Original TabDDPM baseline trained on minority samples only.

    Implements unconditional diffusion with a cosine noise schedule on
    normalized features and clips the generated samples to the [0, 1] range.

    Args:
        X_minority_norm (numpy.ndarray): Normalized minority-class samples of shape (n_samples, n_features).
        n_needed (int): Number of synthetic samples to generate.
        config (dict): Training hyperparameters (e.g., 'lr', 'tabddpm_epochs', 'batch_size').
        seed (int, optional): Random seed for NumPy and PyTorch. Defaults to 42.
        device (str or torch.device, optional): Device to run training and sampling on. Defaults to "cpu".

    Returns:
        numpy.ndarray: Normalized synthetic samples of shape
            (n_needed, n_features). Returns an empty array with shape
            (0, n_features) if there is insufficient data to train.
    """
    if not TABDDPM_AVAILABLE:
        raise RuntimeError("TabDDPM not available")
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    print(f"    Training TabDDPM (Original) on {len(X_minority_norm)} samples...")
    n_features = X_minority_norm.shape[1]
    
    model = MLPDiffusion(
        d_in=n_features, 
        num_classes=0, 
        is_y_cond=False,
        rtdl_params={
            'd_in': n_features, 
            'd_layers': [256, 256],
            'd_out': n_features, 
            'dropout': 0.0
        },
        dim_t=128
    ).to(device)
    
    diffusion = GaussianMultinomialDiffusion(
        num_classes=np.array([0]), 
        num_numerical_features=n_features,
        denoise_fn=model, 
        num_timesteps=1000, 
        gaussian_loss_type='mse',
        scheduler='cosine',
        device=device
    ).to(device)
    
    BASELINE_LR = config.get('lr', 2e-4)
    BASELINE_EPOCHS = config.get('tabddpm_epochs', 600)
    
    optimizer = torch.optim.AdamW(
        diffusion.parameters(), 
        lr=BASELINE_LR, 
        weight_decay=1e-5,
        betas=(0.9, 0.999)
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=BASELINE_EPOCHS,
        eta_min=0.0
    )
    
    diffusion.train()
    X_tensor = torch.FloatTensor(X_minority_norm).to(device)
    batch_size = min(config.get('batch_size', 256), len(X_tensor))
    if batch_size == 0:
        print("    Not enough data to train, skipping.")
        return np.array([]).reshape(0, n_features)

    for epoch in range(BASELINE_EPOCHS):
        perm = torch.randperm(len(X_tensor))
        epoch_loss = 0.0
        n_batches = 0
        
        for i in range(0, len(X_tensor), batch_size):
            indices = perm[i:i+batch_size]
            x_batch = X_tensor[indices]
            
            optimizer.zero_grad()
            out_dict = {}
            loss_multi, loss_gauss = diffusion.mixed_loss(x_batch, out_dict)
            loss = loss_multi.mean() + loss_gauss.mean()
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        
        if (epoch + 1) % 100 == 0 or (BASELINE_EPOCHS < 500 and (epoch + 1) % 50 == 0):
            avg_loss = epoch_loss / (n_batches if n_batches > 0 else 1)
            print(f"        Epoch {epoch+1}/{BASELINE_EPOCHS}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
    
    print(f"    Generating {n_needed} samples...")
    diffusion.eval()
    
    MAX_BATCH_SIZE = 10000
    all_samples = []
    
    with torch.no_grad():
        n_remaining = n_needed
        while n_remaining > 0:
            batch_size_gen = min(MAX_BATCH_SIZE, n_remaining)
            y_dist = torch.ones(batch_size_gen, 1, dtype=torch.float32, device=device)
            sample_output = diffusion.sample(batch_size_gen, y_dist=y_dist)
            
            if isinstance(sample_output, tuple):
                for candidate in sample_output:
                    if isinstance(candidate, torch.Tensor) and candidate.shape == (batch_size_gen, n_features):
                        batch_samples = candidate
                        break
            else:
                batch_samples = sample_output
            
            all_samples.append(batch_samples.cpu().numpy())
            n_remaining -= batch_size_gen
            
            if n_needed > 10000:
                print(f"        Generated {n_needed - n_remaining}/{n_needed}...")
        
        if not all_samples:
             return np.array([]).reshape(0, n_features)
             
        generated_samples_np = np.vstack(all_samples)
    
    generated_samples_np = np.clip(generated_samples_np, 0, 1)
    
    print(f"    Generated shape: {generated_samples_np.shape}")
    print(f"    Value range: [{generated_samples_np.min():.2f}, {generated_samples_np.max():.2f}]")
    
    return generated_samples_np
