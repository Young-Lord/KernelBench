"""
GPU platform abstraction for KernelBench.

Supports NVIDIA CUDA, AMD ROCm (via torch.cuda compatibility), and
Moore Threads MUSA (via torch_musa).
"""

from __future__ import annotations

import os
from typing import Literal

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


def _import_torch_musa():
    import torch_musa  # noqa: F401

    return torch


def get_gpu_platform() -> GpuPlatform | None:
    """Return the active GPU platform, or None if no GPU is available."""
    if torch.cuda.is_available():
        return "cuda"
    try:
        import torch_musa

        if torch_musa.is_available():
            return "musa"
    except ImportError:
        pass
    return None


def is_gpu_available() -> bool:
    return get_gpu_platform() is not None


def get_gpu_module():
    """Return torch.cuda or torch.musa for device operations."""
    platform = get_gpu_platform()
    if platform == "cuda":
        return torch.cuda
    if platform == "musa":
        _import_torch_musa()
        return torch.musa
    raise RuntimeError("No GPU platform available (CUDA or MUSA)")


def get_device_type() -> str:
    platform = get_gpu_platform()
    if platform is None:
        raise RuntimeError("No GPU platform available")
    return platform


def resolve_device(device: torch.device | int | str | None = None) -> torch.device:
    """
    Normalize a device for the current GPU platform.

    Maps cuda:N -> musa:N when running on MUSA hardware.
    """
    device_type = get_device_type()

    if device is None:
        return torch.device(f"{device_type}:0")

    if isinstance(device, int):
        return torch.device(f"{device_type}:{device}")

    if isinstance(device, str):
        device = torch.device(device)

    if device.type == "cuda" and device_type == "musa":
        index = device.index if device.index is not None else 0
        return torch.device(f"musa:{index}")

    if device.type not in ("cuda", "musa"):
        raise ValueError(f"Unsupported device type {device.type!r}; expected cuda or musa")

    if device.index is None:
        return torch.device(f"{device.type}:0")

    return device


def get_gpu_vendor(device: torch.device | int | None = None) -> GpuVendor:
    """Return GPU vendor for the given device."""
    platform = get_gpu_platform()
    if platform is None:
        return "unknown"

    gpu = get_gpu_module()
    if device is None:
        device = gpu.current_device()

    name = gpu.get_device_name(device).upper()
    if "NVIDIA" in name:
        return "nvidia"
    if "AMD" in name or "MI3" in name:
        return "amd"
    if "MTT" in name or "MOORE" in name or "MTHREADS" in name:
        return "musa"
    if platform == "musa":
        return "musa"
    return "unknown"


def _normalize_musa_arch(arch: str) -> str:
    return MUSA_ARCH_ALIASES.get(arch, arch)


def set_gpu_arch(arch_list: list[str]) -> None:
    """
    Set environment variables for kernel compilation on the target architecture.

    Supports NVIDIA (TORCH_CUDA_ARCH_LIST), AMD (PYTORCH_ROCM_ARCH),
    and MUSA (TORCH_MUSA_ARCH / MUSA_ARCH_LIST).
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
            f"Cannot mix NVIDIA, AMD, and MUSA architectures. "
            f"Got NVIDIA={nvidia_archs}, AMD={amd_archs}, MUSA={musa_archs}"
        )

    if nvidia_archs:
        os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(nvidia_archs)
    elif amd_archs:
        os.environ["PYTORCH_ROCM_ARCH"] = ";".join(amd_archs)
    elif musa_archs:
        normalized = musa_archs
        os.environ["MUSA_ARCH_LIST"] = ";".join(normalized)
        # TORCH_MUSA_ARCH uses numeric form, e.g. mp_22 -> 22
        os.environ["TORCH_MUSA_ARCH"] = normalized[-1].replace("mp_", "")


def configure_musa_env() -> None:
    """Ensure MUSA_HOME and related paths are set for extension builds."""
    musa_home = os.environ.get("MUSA_HOME") or "/usr/local/musa"
    os.environ.setdefault("MUSA_HOME", musa_home)
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if musa_home not in ld_path:
        os.environ["LD_LIBRARY_PATH"] = f"{musa_home}/lib:{ld_path}" if ld_path else f"{musa_home}/lib"
    path = os.environ.get("PATH", "")
    if f"{musa_home}/bin" not in path:
        os.environ["PATH"] = f"{musa_home}/bin:{path}" if path else f"{musa_home}/bin"


_MUSA_PATCHED = False


def activate_musa_compat() -> None:
    """
    Patch Tensor/Module .cuda() calls to use musa devices on MUSA hardware.

    Reference KernelBench problems use .cuda() in get_inputs(); this keeps them
    working without modifying every problem file.
    """
    global _MUSA_PATCHED
    if _MUSA_PATCHED or get_gpu_platform() != "musa":
        return

    _import_torch_musa()
    device_type = get_device_type()

    def _to_gpu(self, device=None, non_blocking=False):
        target = resolve_device(device)
        return self.to(device=target, non_blocking=non_blocking)

    torch.Tensor.cuda = _to_gpu
    torch.nn.Module.cuda = lambda self, device=None: self.to(device=resolve_device(device))
    _MUSA_PATCHED = True
