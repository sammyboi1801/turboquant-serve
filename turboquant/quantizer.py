"""
turboquant/quantizer.py — Group-wise min/max quantizer for value compression.

Packs multiple low-bit values into uint8 bytes via vectorized bit-shifting.
Supports 2, 4, and 8 bits per element with configurable group size.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class GroupQuantizer:
    """
    Group-wise affine quantization for KV cache values.

    Each group of `group_size` elements shares a (scale, zero) pair.
    Values are packed into uint8 bytes: 4 values/byte at 2-bit,
    2 values/byte at 4-bit, 1 value/byte at 8-bit.
    """

    def __init__(self, bits: int = 4, group_size: int = 32):
        assert bits in (2, 4, 8), "bits must be 2, 4, or 8"
        self.bits = bits
        self.gs = group_size
        self.maxq = (1 << bits) - 1
        self.ppb = 8 // bits  # packed values per byte

    @torch.no_grad()
    def quantize(self, x: torch.Tensor):
        """
        Quantize x along the last dimension.

        Returns:
            packed  — uint8 tensor of packed indices
            scale   — float tensor, shape (..., n_groups)
            zero    — float tensor, shape (..., n_groups)
            meta    — dict with padding info needed for dequantize
        """
        D = x.shape[-1]
        prefix = x.shape[:-1]
        pad_d = (self.gs - D % self.gs) % self.gs
        xp = F.pad(x, (0, pad_d)) if pad_d else x
        Dp = xp.shape[-1]
        ng = Dp // self.gs

        xg = xp.reshape(*prefix, ng, self.gs)
        mn = xg.amin(dim=-1, keepdim=True)
        mx = xg.amax(dim=-1, keepdim=True)
        scale = ((mx - mn) / self.maxq).clamp(min=1e-8)
        zero = mn
        q = ((xg - zero) / scale).round().clamp(0, self.maxq).to(torch.uint8)

        q_flat = q.reshape(*prefix, Dp)
        ppb = self.ppb
        pad_p = (ppb - Dp % ppb) % ppb
        if pad_p:
            q_flat = F.pad(q_flat, (0, pad_p))

        # Vectorized packing: reshape to groups of ppb, shift, sum
        q_rs = q_flat.view(*prefix, -1, ppb).to(torch.int16)
        shifts = torch.arange(0, self.bits * ppb, self.bits, device=x.device)
        packed = (q_rs << shifts).sum(dim=-1).to(torch.uint8)

        meta = {"D": D, "Dp": Dp, "pad_d": pad_d, "pad_p": pad_p}
        return packed, scale.squeeze(-1), zero.squeeze(-1), meta

    @torch.no_grad()
    def dequantize(self, packed, scale, zero, meta) -> torch.Tensor:
        """Reconstruct float tensor from packed quantized values."""
        D, Dp = meta["D"], meta["Dp"]
        pad_d, pad_p = meta["pad_d"], meta["pad_p"]
        ppb, bits, mask = self.ppb, self.bits, (1 << self.bits) - 1
        prefix = packed.shape[:-1]

        # Vectorized unpacking: broadcast shifts over packed bytes
        shifts = torch.arange(0, bits * ppb, bits, device=packed.device)
        q_flat = (packed.unsqueeze(-1).to(torch.int16) >> shifts) & mask
        q_flat = q_flat.flatten(-2, -1).to(torch.uint8)
        if pad_p:
            q_flat = q_flat[..., :-pad_p]

        xg = q_flat.reshape(*prefix, scale.shape[-1], -1).to(scale.dtype)
        xr = (xg * scale.unsqueeze(-1) + zero.unsqueeze(-1)).reshape(*prefix, Dp)
        return xr[..., :-pad_d] if pad_d else xr
