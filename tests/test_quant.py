import torch
from turboquant import LloydMaxCodebook, GroupQuantizer
from turboquant.cache import TurboQuantCache

def test_codebook_roundtrip():
    cb = LloydMaxCodebook(bits=4, head_dim=64, n_samples=5000, n_iter=20)
    x = torch.randn(8, 4, 32, 64)
    mag = x.norm(dim=-1)
    xu = x / mag.unsqueeze(-1).clamp(min=1e-8)
    idx = cb.encode(xu)
    xr = cb.decode(idx)
    cos = torch.nn.functional.cosine_similarity(xu.reshape(-1, 64), xr.reshape(-1, 64), dim=-1).mean()
    assert cos > 0.90, f"codebook cos_sim too low: {cos:.4f}"

def test_group_quantizer_roundtrip():
    gq = GroupQuantizer(bits=4, group_size=32)
    x = torch.randn(2, 8, 16, 64)
    packed, scale, zero, meta = gq.quantize(x)
    xr = gq.dequantize(packed, scale, zero, meta)
    assert xr.shape == x.shape
    rel_err = (x - xr).abs().mean() / x.abs().mean()
    assert rel_err < 0.15, f"group quant error too high: {rel_err:.4f}"

def test_cache_update_decode():
    cache = TurboQuantCache(key_bits=4, value_bits=4, device="cpu", dtype=torch.float32)
    k = torch.randn(1, 4, 16, 64)
    v = torch.randn(1, 4, 16, 64)
    k_out, v_out = cache.update(k, v, layer_idx=0)
    assert k_out.shape == k.shape
    assert v_out.shape == v.shape
    assert cache.get_seq_length(0) == 16

def test_cache_accumulates():
    cache = TurboQuantCache(key_bits=4, value_bits=4, device="cpu", dtype=torch.float32)
    for i in range(5):
        k = torch.randn(1, 4, 1, 64)
        v = torch.randn(1, 4, 1, 64)
        k_out, v_out = cache.update(k, v, layer_idx=0)
        assert k_out.shape[2] == i + 1
    assert cache.get_seq_length(0) == 5
