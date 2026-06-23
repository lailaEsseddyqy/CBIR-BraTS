# src/training/supcon_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.tau = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = embeddings.device
        valid  = labels != -1
        if valid.sum() < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        z = embeddings[valid]
        y = labels[valid]
        B = z.shape[0]

        sim = torch.mm(z, z.T) / self.tau

        self_mask = torch.eye(B, dtype=torch.bool, device=device)
        # CORRECTION : -1e9 au lieu de float("-inf")
        # évite le piège 0 * (-inf) = NaN lors du masquage par pos_mask
        sim = sim.masked_fill(self_mask, -1e9)

        label_eq = (y.unsqueeze(0) == y.unsqueeze(1))
        pos_mask = label_eq & ~self_mask

        has_pos = pos_mask.sum(dim=1) > 0
        if has_pos.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)

        pos_count = pos_mask.sum(dim=1).clamp(min=1)
        loss_per_anchor = -(log_prob * pos_mask).sum(dim=1) / pos_count

        return loss_per_anchor[has_pos].mean()