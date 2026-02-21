"""
Fisher Information geometry for multi-class manifolds.
Provides class-specific curvature to guide augmentation allocation.
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy.linalg import eigh

class FisherInformationCalculator:
    """
    Compute Fisher Information Matrix (FIM) and scalar curvature
    for multi-class tabular distributions.
    """
    
    def __init__(self, n_classes, n_features, device='cpu'):
        self.n_classes = n_classes
        self.n_features = n_features
        self.device = device
        self.class_curvatures = {}
    
    def compute_fim(self, X, y, model):
        """
        Compute Fisher Information Matrix using log-likelihood gradients.
        
        FIM[i,j] = E[∂log p(x|θ)/∂θ_i × ∂log p(x|θ)/∂θ_j]
        """
        model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y, dtype=torch.long, device=self.device)
        
        # Compute log probabilities
        logits = model(X_tensor)
        log_probs = F.log_softmax(logits, dim=1)
        
        # Extract per-sample gradients
        gradients = []
        for i in range(len(X)):
            model.zero_grad()
            log_prob = log_probs[i, y_tensor[i]]
            log_prob.backward(retain_graph=True)
            
            # Flatten all parameter gradients
            grad_vec = torch.cat([
                p.grad.flatten() for p in model.parameters() if p.grad is not None
            ])
            gradients.append(grad_vec.cpu().numpy())
        
        gradients = np.array(gradients)
        
        # FIM = E[g × g^T]
        fim = gradients.T @ gradients / len(X)
        
        return fim
    
    def batch_approximated_curvature(self, X, y, model, batch_size=256):
        """
        Memory-efficient curvature estimation via stochastic batches.
        Uses trace of FIM as scalar curvature proxy.
        
        Returns:
            dict: {class_id: curvature_score}
        """
        curvatures = {}
        
        for class_id in range(self.n_classes):
            class_mask = (y == class_id)
            X_class = X[class_mask]
            
            if len(X_class) < 10:
                curvatures[class_id] = 0.0
                continue
            
            # Subsample for efficiency
            n_samples = min(len(X_class), batch_size)
            indices = np.random.choice(len(X_class), n_samples, replace=False)
            X_batch = X_class[indices]
            y_batch = np.full(n_samples, class_id)
            
            # Compute FIM for this class
            fim = self.compute_fim(X_batch, y_batch, model)
            
            # Scalar curvature = trace(FIM)
            curvature = np.trace(fim)
            curvatures[class_id] = curvature
        
        self.class_curvatures = curvatures
        return curvatures
    
    def get_augmentation_weights(self, class_counts):
        """
        Allocate augmentation budget proportional to curvature × rarity.
        
        w_c = (κ_c / Σκ) × (1 / n_c)
        """
        if not self.class_curvatures:
            raise ValueError("Must call batch_approximated_curvature first")
        
        weights = {}
        total_curvature = sum(self.class_curvatures.values())
        
        for class_id, count in class_counts.items():
            curvature = self.class_curvatures.get(class_id, 1.0)
            rarity = 1.0 / (count + 1e-6)
            
            # Normalize by total curvature
            weight = (curvature / total_curvature) * rarity
            weights[class_id] = weight
        
        # Normalize to sum=1
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        
        return weights
