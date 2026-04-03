"""
turboquant — KV-cache compression for HuggingFace transformers.

Drop-in replacement for DynamicCache:
    from turboquant import TurboQuantCache
    cache = TurboQuantCache(key_bits=4, value_bits=4)
    outputs = model.generate(**inputs, past_key_values=cache)
"""

from turboquant.cache import TurboQuantCache
from turboquant.codebook import LloydMaxCodebook
from turboquant.quantizer import GroupQuantizer
from turboquant.utils import load_model, get_dims

__all__ = [
    "TurboQuantCache",
    "LloydMaxCodebook",
    "GroupQuantizer",
    "load_model",
    "get_dims",
]

__version__ = "0.1.1"