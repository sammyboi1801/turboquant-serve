"""
turboquant/cache.py: TurboQuantCache, compressed KV cache for HuggingFace models.

Drop-in replacement for DynamicCache. Pass to model.generate() and tokens are
compressed on the fly with no changes to model code.

Algorithm:
  Keys:   rotate (random orthogonal Q) → L2-normalize → Lloyd-Max encode → bit-pack
  Values: rotate → group-wise min/max quantize → bit-pack

No QJL correction - omitted intentionally. QJL hurts attention quality because
softmax amplifies its variance. MSE-only quantization has lower variance and wins.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import DynamicCache

from turboquant.codebook import LloydMaxCodebook
from turboquant.quantizer import GroupQuantizer


class TurboQuantCache(DynamicCache):
    """
    Subclasses DynamicCache and overrides update() to compress K/V on the fly.
    model.generate() calls this without any changes to model code.
    """

    def __init__(
        self,
        key_bits: int = 4,
        value_bits: int = 4,
        group_size: int = 32,
        device: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self._key_bits = key_bits
        self._dtype = dtype
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._val_q = GroupQuantizer(bits=value_bits, group_size=group_size)
        self.verbose = True          # set False after warmup to silence mid-request prints
        # Lazy per-dimension caches
        self._R: dict[int, torch.Tensor] = {}
        self._cb: dict[int, LloydMaxCodebook] = {}
        # Compressed stores
        self._ks: dict[int, dict] = {}
        self._vs: dict[int, dict] = {}
        self._seq: dict[int, int] = {}

    def _rotation(self, d: int) -> torch.Tensor:
        if d not in self._R:
            torch.manual_seed(d)
            Q, _ = torch.linalg.qr(
                torch.randn(d, d, device=self._device, dtype=torch.float32))
            self._R[d] = Q.to(self._dtype)
        return self._R[d]

    def _codebook(self, d: int) -> LloydMaxCodebook:
        if d not in self._cb:
            if self.verbose:
                print(f"  [TQ] Fitting codebook bits={self._key_bits} dim={d}…")
            cb = LloydMaxCodebook(bits=self._key_bits, head_dim=d,
                                  device=self._device)
            cb.centroids = cb.centroids.to(self._device)
            self._cb[d] = cb
        return self._cb[d]

    @staticmethod
    def _pack_indices(idx: torch.Tensor, bits: int) -> torch.Tensor:
        """Vectorized bit-packing. Requires bits in {2, 4, 8}."""
        if bits == 8:
            return idx
        assert 8 % bits == 0, "bits must divide 8 evenly (use 2, 4, or 8)"
        ppb = 8 // bits
        *prefix, D = idx.shape
        pad = (ppb - D % ppb) % ppb
        x = F.pad(idx.to(torch.int16), (0, pad)) if pad else idx.to(torch.int16)
        shifts = torch.arange(0, bits * ppb, bits, device=idx.device)
        packed = (x.view(*prefix, -1, ppb) << shifts).sum(dim=-1).to(torch.uint8)
        return packed

    @staticmethod
    def _unpack_indices(packed: torch.Tensor, bits: int, D: int) -> torch.Tensor:
        """Vectorized bit-unpacking. Returns (..., D) uint8."""
        if bits == 8:
            return packed
        ppb = 8 // bits
        mask = (1 << bits) - 1
        shifts = torch.arange(0, bits * ppb, bits, device=packed.device)
        unpacked = (packed.unsqueeze(-1).to(torch.int16) >> shifts) & mask
        return unpacked.flatten(-2, -1).to(torch.uint8)[..., :D]

    @torch.no_grad()
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs=None,
    ):
        k = key_states.to(self._dtype)
        v = value_states.to(self._dtype)
        D = k.shape[-1]
        R = self._rotation(D)
        cb = self._codebook(D)

        # Rotate
        kr = k @ R
        vr = v @ R

        # Quantize keys: magnitude + bit-packed Lloyd-Max indices
        mag = kr.norm(dim=-1)
        ku  = kr / mag.unsqueeze(-1).clamp(min=1e-8)
        ki  = cb.encode(ku)                             # (..., D) uint8
        kip = self._pack_indices(ki, self._key_bits)    # bit-packed

        # Quantize values: group quant
        vp, vs, vz, vm = self._val_q.quantize(vr)

        if layer_idx not in self._ks:
            self._ks[layer_idx] = {"i": [], "m": [], "D": D, "vm": vm}
            self._vs[layer_idx] = {"p": [], "s": [], "z": []}
            self._seq[layer_idx] = 0

        self._ks[layer_idx]["i"].append(kip)
        self._ks[layer_idx]["m"].append(mag)
        self._vs[layer_idx]["p"].append(vp)
        self._vs[layer_idx]["s"].append(vs)
        self._vs[layer_idx]["z"].append(vz)
        self._seq[layer_idx] += k.shape[2]

        return self._decode(layer_idx)

    def _decode(self, li: int):
        ks = self._ks[li]
        vs = self._vs[li]
        D  = ks["D"]
        R  = self._rotation(D)
        cb = self._codebook(D)

        kip = torch.cat(ks["i"], dim=2)
        km  = torch.cat(ks["m"], dim=2)
        vp  = torch.cat(vs["p"], dim=2)
        vsc = torch.cat(vs["s"], dim=2)
        vz  = torch.cat(vs["z"], dim=2)

        # Unpack + decode keys
        ki    = self._unpack_indices(kip, self._key_bits, D)
        ku    = cb.decode(ki).to(self._dtype)
        k_out = (ku * km.to(self._dtype).unsqueeze(-1)) @ R.T

        # Decode values
        v_out = self._val_q.dequantize(vp, vsc, vz, ks["vm"]).to(self._dtype) @ R.T

        return k_out, v_out

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._seq.get(layer_idx, 0)

    def bytes_used(self) -> int:
        """Bytes used by the compressed store (packed key indices + value quant)."""
        t = 0
        for s in self._ks.values():
            for x in s["i"]: t += x.numel() * x.element_size()
            for x in s["m"]: t += x.numel() * x.element_size()
        for s in self._vs.values():
            for x in s["p"]: t += x.numel() * x.element_size()
            for x in s["s"]: t += x.numel() * x.element_size()
            for x in s["z"]: t += x.numel() * x.element_size()
        return t

    def compression_stats(self, baseline_dtype_bytes: int = 2) -> dict:
        """
        Compare compressed store size vs what a standard fp16/bf16 cache would use.
        baseline_dtype_bytes=2 for bf16/fp16, 4 for fp32.
        """
        compressed = self.bytes_used()
        baseline = 0
        for li, ks in self._ks.items():
            T = self._seq.get(li, 0)
            D = ks["D"]
            if ks["m"]:
                H = ks["m"][0].shape[1]  # (B, H, S) → H
                baseline += 2 * T * H * D * baseline_dtype_bytes  # K + V
        return {
            "compressed_MB": compressed / 1e6,
            "baseline_MB":   baseline / 1e6,
            "ratio":         baseline / max(compressed, 1),
            "key_bits":      self._key_bits,
        }
