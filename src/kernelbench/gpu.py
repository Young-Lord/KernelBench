"""
GPU platform abstraction for KernelBench — powered by torchada.

Supports NVIDIA CUDA and Moore Threads MUSA via a single unified API.
torchada transparently routes torch.cuda.* calls to torch.musa.* on MUSA hardware,
eliminating the need for hand-written device-resolution, monkey-patching, or
environment-configuration code.

Usage:
    import kernelbench.gpu as gpu

    if gpu.is_gpu_available():
        device = gpu.resolve_device("cuda:0")
        # Use device with any torch.cuda.* API — torchada handles MUSA routing
"""

from __future__ import annotations

import os
from typing import Literal

import torchada  # noqa: F401 — Apply patches so torch.cuda.* works on MUSA
import torch

GpuPlatform = Literal["cuda", "musa"]
GpuVendor = Literal["nvidia", "amd", "musa", "unknown"]

NVIDIA_ARCHS = ["Maxwell", "Pascal", "Volta", "Turing", "Ampere", "Hopper", "Ada", "Blackwell"]
AMD_ARCHS = ["gfx942", "gfx950"]
MUSA_ARCHS = ["mp_21", "mp_22", "mp_31", "S80", "S3000", "S4000", "S5000"]

# Friendly GPU name -> mcc arch flag
MUSA_ARCH_ALIASES = {
    "S80": "mp_21",
    "S3000": "mp_21",
    "S4000": "mp_22",
    "S5000": "mp_31",
}


# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------

def get_gpu_platform() -> GpuPlatform | None:
    """Return the active GPU platform ('cuda' or 'musa'), or None."""
    platform = torchada.get_platform()
    if platform == torchada.Platform.CUDA:
        return "cuda"
    if platform == torchada.Platform.MUSA:
        return "musa"
    return None


def is_gpu_available() -> bool:
    """Return True if any GPU (NVIDIA CUDA or Moore Threads MUSA) is available.

    Uses torchada.cuda.is_available() instead of torch.cuda.is_available()
    because torchada intentionally does NOT patch torch.cuda.is_available()
    (it would break CUDA-specific tooling that expects the original value).
    """
    return torchada.cuda.is_available()


def get_gpu_module():
    """
    Return the torch.cuda module.

    torchada patches torch.cuda at import time so that all device operations
    transparently route to torch.musa when running on MUSA hardware.  Callers
    can use the returned module as if it were vanilla torch.cuda.
    """
    return torch.cuda


def get_device_type() -> str:
    """Return the device type string ('cuda' or 'musa')."""
    platform = get_gpu_platform()
    if platform is None:
        raise RuntimeError("No GPU platform available (CUDA or MUSA)")
    return platform


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------

def resolve_device(device: torch.device | int | str | None = None) -> torch.device:
    """
    Normalize a device reference for the current GPU platform.

    With torchada, CUDA devices are automatically routed to MUSA at the API
    level, so we only need to fill in defaults and validate the type.
    """
    fallback_type = get_gpu_platform() or "cuda"

    if device is None:
        return torch.device(f"{fallback_type}:0")

    if isinstance(device, int):
        return torch.device(f"{fallback_type}:{device}")

    if isinstance(device, str):
        device = torch.device(device)

    if device.type not in ("cuda", "musa"):
        raise ValueError(
            f"Unsupported device type {device.type!r}; expected 'cuda' or 'musa'"
        )

    if device.index is None:
        gpu_platform = get_gpu_platform()
        device_type = gpu_platform if gpu_platform else device.type
        return torch.device(f"{device_type}:0")

    return device


# ---------------------------------------------------------------------------
# Vendor identification
# ---------------------------------------------------------------------------

def get_gpu_vendor(device: torch.device | int | None = None) -> GpuVendor:
    """Return the GPU vendor for the given (or current) device."""
    platform = get_gpu_platform()
    if platform is None:
        return "unknown"

    if device is None:
        device = torch.cuda.current_device()

    name = torch.cuda.get_device_name(device).upper()
    if "NVIDIA" in name:
        return "nvidia"
    if "AMD" in name or "MI3" in name:
        return "amd"
    if "MTT" in name or "MOORE" in name or "MTHREADS" in name:
        return "musa"
    if platform == "musa":
        return "musa"
    return "unknown"


# ---------------------------------------------------------------------------
# Architecture configuration for kernel compilation
# ---------------------------------------------------------------------------

def _normalize_musa_arch(arch: str) -> str:
    return MUSA_ARCH_ALIASES.get(arch, arch)


def set_gpu_arch(arch_list: list[str]) -> None:
    """
    Set environment variables for kernel compilation targeting specific GPU architectures.

    Supports NVIDIA (TORCH_CUDA_ARCH_LIST), AMD (PYTORCH_ROCM_ARCH),
    and MUSA (MUSA_ARCH_LIST / TORCH_MUSA_ARCH).
    """
    nvidia_archs: list[str] = []
    amd_archs: list[str] = []
    musa_archs: list[str] = []

    for arch in arch_list:
        if arch in NVIDIA_ARCHS:
            nvidia_archs.append(arch)
        elif arch in AMD_ARCHS:
            amd_archs.append(arch)
        elif arch in MUSA_ARCHS:
            musa_archs.append(_normalize_musa_arch(arch))
        else:
            raise ValueError(
                f"Invalid architecture: {arch}. Must be one of "
                f"NVIDIA: {NVIDIA_ARCHS}, AMD: {AMD_ARCHS}, MUSA: {MUSA_ARCHS}"
            )

    configured = sum(bool(x) for x in (nvidia_archs, amd_archs, musa_archs))
    if configured > 1:
        raise ValueError(
            f"Cannot mix architectures from different vendors. "
            f"Got NVIDIA={nvidia_archs}, AMD={amd_archs}, MUSA={musa_archs}"
        )

    if nvidia_archs:
        os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(nvidia_archs)
    elif amd_archs:
        os.environ["PYTORCH_ROCM_ARCH"] = ";".join(amd_archs)
    elif musa_archs:
        normalized = musa_archs
        os.environ["MUSA_ARCH_LIST"] = ";".join(normalized)
        # TORCH_MUSA_ARCH uses the numeric form, e.g. mp_22 -> 22
        os.environ["TORCH_MUSA_ARCH"] = normalized[-1].replace("mp_", "")
