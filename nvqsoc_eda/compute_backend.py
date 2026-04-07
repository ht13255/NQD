from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any


@dataclass(frozen=True, slots=True)
class ComputeBackendInfo:
    engine: str
    device: str
    gpu_enabled: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


@lru_cache(maxsize=1)
def get_torch_module() -> Any | None:
    try:
        import torch  # type: ignore[import-not-found]
    except Exception:
        return None
    return torch


@lru_cache(maxsize=1)
def resolve_compute_backend() -> ComputeBackendInfo:
    if _env_flag("NVQSOC_FORCE_CPU", False):
        return ComputeBackendInfo(engine="numpy", device="cpu", gpu_enabled=False, reason="forced_cpu")
    if not _env_flag("NVQSOC_ENABLE_GPU", True):
        return ComputeBackendInfo(engine="numpy", device="cpu", gpu_enabled=False, reason="gpu_disabled")

    torch = get_torch_module()
    if torch is None:
        return ComputeBackendInfo(engine="numpy", device="cpu", gpu_enabled=False, reason="torch_unavailable")

    if torch.cuda.is_available():
        return ComputeBackendInfo(engine="torch", device="cuda", gpu_enabled=True, reason="cuda_available")

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return ComputeBackendInfo(engine="torch", device="mps", gpu_enabled=True, reason="mps_available")

    return ComputeBackendInfo(engine="numpy", device="cpu", gpu_enabled=False, reason="no_gpu_backend")
