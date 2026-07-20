"""
MUSA inline extension loader for KernelBench.

Provides a load_inline-compatible API backed by torch_musa MUSAExtension,
since standard torch.utils.cpp_extension.load_inline requires CUDA_HOME/nvcc.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Sequence

from kernelbench.gpu import configure_musa_env


def _make_module_name(name: str) -> str:
    safe = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
    if not safe or safe[0].isdigit():
        safe = f"ext_{safe}"
    return safe


def _write_setup_py(build_dir: Path, module_name: str, source_name: str) -> None:
    setup_py = build_dir / "setup.py"
    setup_py.write_text(
        f"""\
from setuptools import setup
from torch_musa.utils.musa_extension import MUSAExtension, BuildExtension

setup(
    name="{module_name}",
    ext_modules=[
        MUSAExtension(
            name="{module_name}",
            sources=["{source_name}"],
        )
    ],
    cmdclass={{"build_ext": BuildExtension}},
)
"""
    )


def _build_pybind_block(functions: Sequence[str]) -> str:
    bindings = "\n    ".join(
        f'm.def("{fn}", &{fn}, "{fn}");' for fn in functions
    )
    return f"""
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    {bindings}
}}
"""


def load_inline(
    name: str,
    cpp_sources: str | Sequence[str] | None = None,
    cuda_sources: str | Sequence[str] | None = None,
    functions: Sequence[str] | None = None,
    verbose: bool = False,
    extra_cflags=None,
    extra_ldflags=None,
    with_cuda: bool = True,
    **kwargs,
):
    """
    JIT-compile and load a MUSA extension, mirroring torch.utils.cpp_extension.load_inline.

    Accepts kernel code in cpp_sources and/or cuda_sources. The combined source is
    compiled as a .mu file with mcc via MUSAExtension.
    """
    del extra_cflags, extra_ldflags, with_cuda, kwargs  # reserved for API compatibility

    if not functions:
        raise ValueError("load_inline requires at least one function name")

    configure_musa_env()

    parts: list[str] = []
    for src in (cpp_sources, cuda_sources):
        if src is None:
            continue
        if isinstance(src, (list, tuple)):
            parts.extend(src)
        else:
            parts.append(src)

    if not parts:
        raise ValueError("load_inline requires cpp_sources and/or cuda_sources")

    source_body = "\n".join(parts)
    if "PYBIND11_MODULE" not in source_body:
        source_body = source_body.rstrip() + _build_pybind_block(functions)

    module_name = _make_module_name(name)
    source_hash = hashlib.sha256(source_body.encode()).hexdigest()[:16]
    build_root = Path(
        os.environ.get(
            "TORCH_EXTENSIONS_DIR",
            os.path.join(os.path.expanduser("~"), ".cache", "torch_extensions"),
        )
    )
    build_dir = build_root / f"musa_{module_name}_{source_hash}"
    build_dir.mkdir(parents=True, exist_ok=True)

    source_path = build_dir / "kernel.mu"
    if not source_path.exists() or source_path.read_text() != source_body:
        source_path.write_text(source_body)

    _write_setup_py(build_dir, module_name, source_path.name)

    so_candidates = list(build_dir.glob(f"{module_name}*.so"))
    if not so_candidates:
        import subprocess

        cmd = [sys.executable, "setup.py", "build_ext", "--inplace"]
        if verbose:
            print(f"[musa_extension] Building {module_name} in {build_dir}")
        result = subprocess.run(
            cmd,
            cwd=build_dir,
            capture_output=not verbose,
            text=True,
        )
        if result.returncode != 0:
            msg = result.stderr or result.stdout or "unknown build error"
            raise RuntimeError(f"MUSA extension build failed for {module_name}:\n{msg}")
        so_candidates = list(build_dir.glob(f"{module_name}*.so"))

    if not so_candidates:
        raise RuntimeError(f"MUSA extension build produced no .so for {module_name}")

    so_path = so_candidates[0]
    spec = importlib.util.spec_from_file_location(module_name, so_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load MUSA extension module from {so_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
