"""PyPTO-FlashKDA — Kimi Delta Attention kernels for Ascend 910C (PyPTO)."""

from .api import available, bwd, fwd

__version__ = "0.1.0"
__all__ = ["fwd", "bwd", "available"]
