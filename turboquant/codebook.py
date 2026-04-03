"""
turboquant/codebook.py: Lloyd-Max scalar codebook for key quantization.

Uses a Beta distribution prior fitted to rotated, unit-normalized key vectors.
MSE-optimal quantization. QJL correction intentionally omitted (hurts attention
quality because softmax amplifies its variance).
"""

from __future__ import annotations

import torch


class LloydMaxCodebook:
    """
    Lloyd-Max scalar quantizer with Beta distribution prior.

    The Beta(alpha, alpha) distribution approximates the marginal distribution
    of entries in rotated, L2-normalized vectors. alpha is derived from head_dim
    so the prior tightens as dimension grows (entries concentrate near zero).
    """

    def __init__(
        self,
        bits: int,
        head_dim: int,
        n_samples: int = 200_000,
        n_iter: int = 80,
        device: str = "cpu",
    ):
        self.bits = bits
        self.n_levels = 2 ** bits
        alpha = max((head_dim - 1) / 2.0, 1.01)
        self.centroids = self._fit(alpha, n_samples, n_iter, device)

    def _fit(self, alpha: float, n: int, iters: int, device: str) -> torch.Tensor:
        data = (torch.distributions.Beta(alpha, alpha).sample((n,)) * 2 - 1).to(device)
        c = torch.linspace(data.min().item(), data.max().item(),
                           self.n_levels, device=device)
        for _ in range(iters):
            idx = (data.unsqueeze(1) - c.unsqueeze(0)).abs().argmin(dim=1)
            new_c = c.clone()
            for k in range(self.n_levels):
                sel = data[idx == k]
                if sel.numel() > 0:
                    new_c[k] = sel.mean()
            if (new_c - c).abs().max() < 1e-7:
                break
            c = new_c
        return c.sort().values.contiguous()

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize x to nearest centroid index. Returns uint8 tensor."""
        flat = x.reshape(-1)
        c = self.centroids.to(flat.device)
        idx = (flat.unsqueeze(1) - c.unsqueeze(0)).abs().argmin(dim=1).to(torch.uint8)
        return idx.reshape(x.shape)

    @torch.no_grad()
    def decode(self, idx: torch.Tensor) -> torch.Tensor:
        """Reconstruct float values from centroid indices."""
        return self.centroids.to(idx.device)[idx.long()]
