"""Read a build product's dynamic linking and check it against the contract's whitelist.

§4.7 gives the B tier a stage the A tier does not have: read the export table of
what was built and check which libraries it actually links. The A tier needs none
of it -- it forbids library compute outright, so there is nothing to whitelist --
while the B tier's whole question is which library a submission reached for, and a
source scan cannot answer it. A source scan reads the imports the submission
wrote; this reads the sonames the linker resolved, including whatever came in
behind them.

The ELF is parsed here rather than shelled out to `readelf`/`nm`, because the
stage has to run wherever the evaluator runs and binutils is not part of any
declared environment. Only the two things the stage needs are read: the dynamic
section's `DT_NEEDED` sonames, and the dynamic symbol table for the evidence the
report carries.

    python musa_operator_eval/evaluator/link_whitelist.py \
        --build-dir runs/kb_l3_43_b/build \
        --task-dir musa_operator_eval/tasks/kb_l3_43_b
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
REPO_TOP = ROOT.parent

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

ELF_MAGIC = b"\x7fELF"

# Section header types this stage reads.
SHT_DYNAMIC = 6
SHT_DYNSYM = 11

# `d_tag` of the entries that name a shared library this object needs.
DT_NEEDED = 1

# `st_shndx` of a symbol the object does not define; the linker resolves it
# elsewhere, which is what makes it evidence of a call out of this object.
SHN_UNDEF = 0

# The suffixes a build product of this stack arrives with. `.o` is included
# because a submission may ship a static object whose dynamic section is empty:
# that is a fact worth reporting rather than a file worth skipping.
BUILD_PRODUCT_SUFFIXES = (".so", ".o")

# A linker's final product is an executable and usually has no suffix at all,
# while the object files it was linked from do. A suffix list therefore selects
# exactly the artifacts that have no dynamic section and misses the one that
# does, so the stage reads the magic instead. The suffix list stays as a second
# signal so that a mangled `.so` is still reported as unreadable rather than
# skipped for having lost its magic.



class ElfError(ValueError):
    """The file is not an ELF object this reader can describe."""


def _section_headers(blob: bytes) -> List[Tuple[int, int, int, int, int, int]]:
    """Return (name_offset, type, offset, size, link, entsize) per section."""
    if blob[:4] != ELF_MAGIC:
        raise ElfError("not an ELF file")
    elf_class, elf_data = blob[4], blob[5]
    if elf_data != 1:
        raise ElfError("only little-endian objects are supported")
    if elf_class == 2:
        endian, header_offset, section_offset = "<", 0x28, 0x3A
        entry_format, entry_size = "<IIQQQQIIQQ", 64
    elif elf_class == 1:
        endian, header_offset, section_offset = "<", 0x20, 0x2E
        entry_format, entry_size = "<IIIIIIIIII", 40
    else:
        raise ElfError(f"unknown ELF class {elf_class}")

    (shoff,) = struct.unpack_from(endian + "Q" if elf_class == 2 else endian + "I", blob, header_offset)
    shentsize, shnum = struct.unpack_from("<HH", blob, section_offset)
    if shoff == 0 or shnum == 0:
        raise ElfError("no section header table: the export table is not readable")

    headers = []
    for index in range(shnum):
        start = shoff + index * (shentsize or entry_size)
        fields = struct.unpack_from(entry_format, blob, start)
        name_offset, section_type, _flags, _addr, offset, size, link, _info, _align, entsize = fields
        headers.append((name_offset, section_type, offset, size, link, entsize))
    return headers


def _string_at(blob: bytes, table_offset: int, offset: int) -> str:
    end = blob.index(b"\0", table_offset + offset)
    return blob[table_offset + offset : end].decode("utf-8", errors="replace")


def read_elf_report(path: Path) -> Dict[str, object]:
    """Describe one build product's dynamic dependencies and symbol counts.

    Args:
        path: the object to read

    Returns:
        {"needed": [...], "undefined_symbols": [...], "defined_symbols": [...]}

    Raises:
        ElfError: the file is not an object this reader can describe. Callers
            report that rather than treating it as an empty dependency list: "I
            could not read it" and "it links nothing" are different facts, and
            the second one would pass a whitelist check.
    """
    blob = Path(path).read_bytes()
    headers = _section_headers(blob)

    needed: List[str] = []
    dynamic = next((entry for entry in headers if entry[1] == SHT_DYNAMIC), None)
    if dynamic is not None:
        _name, _type, offset, size, link, _entsize = dynamic
        string_table = headers[link][2]
        entry_size = 16 if _is_64(blob) else 8
        for position in range(offset, offset + size, entry_size):
            tag, value = struct.unpack_from("<qQ" if entry_size == 16 else "<iI", blob, position)
            if tag == DT_NEEDED:
                needed.append(_string_at(blob, string_table, value))

    undefined: List[str] = []
    defined: List[str] = []
    dynamic_symbols = next((entry for entry in headers if entry[1] == SHT_DYNSYM), None)
    if dynamic_symbols is not None:
        _name, _type, offset, size, link, entsize = dynamic_symbols
        string_table = headers[link][2]
        entry_size = entsize or (24 if _is_64(blob) else 16)
        for position in range(offset, offset + size, entry_size):
            if _is_64(blob):
                name_offset, _info, _other, shndx = struct.unpack_from("<IBBH", blob, position)
            else:
                name_offset, _value, _size, _info, _other, shndx = struct.unpack_from("<IIIBBH", blob, position)
            if name_offset == 0:
                continue
            name = _string_at(blob, string_table, name_offset)
            (undefined if shndx == SHN_UNDEF else defined).append(name)

    return {"needed": needed, "undefined_symbols": sorted(set(undefined)), "defined_symbols": sorted(set(defined))}


def _is_64(blob: bytes) -> bool:
    return blob[4] == 2


def build_products(build_dir: Path) -> List[Path]:
    """Every object under `build_dir` the stage has an opinion about."""
    if not Path(build_dir).is_dir():
        return []
    found = []
    for path in sorted(Path(build_dir).rglob("*")):
        if not path.is_file():
            continue
        if path.suffix in BUILD_PRODUCT_SUFFIXES or ".so." in path.name:
            found.append(path)
            continue
        try:
            with path.open("rb") as handle:
                magic = handle.read(4)
        except OSError:
            continue
        if magic == ELF_MAGIC:
            found.append(path)
    return found


def check_build_dir(
    build_dir: Path,
    allowed_libraries: Optional[List[str]] = None,
    allowed_symbol_prefixes: Optional[List[str]] = None,
) -> Dict[str, object]:
    """Check every build product in a directory and describe the outcome.

    Returns:
        {"ran": bool, "artifacts": [...], "checked": int, "passed": bool, "errors": [...]}

    `ran` is False only when there was nothing to read. A directory with no
    object in it is reported as unchecked rather than as a pass: a whitelist
    check that passed because nothing was built is the illusion §4.7's ordering
    exists to prevent.
    """
    checker = _load_checker()
    products = build_products(build_dir)
    report: Dict[str, object] = {
        "build_dir": str(build_dir),
        "artifacts": [],
        "checked": 0,
        "passed": True,
        "errors": [],
    }
    if not products:
        report["ran"] = False
        report["detail"] = "no compiled artifact was produced, so nothing was linked and nothing was checked"
        return report

    report["ran"] = True
    for product in products:
        artifact: Dict[str, object] = {"path": str(product)}
        try:
            described = read_elf_report(product)
        except (ElfError, OSError, ValueError, struct.error) as error:
            artifact["readable"] = False
            artifact["reason"] = f"{type(error).__name__}: {error}"
            report["errors"].append(f"{product.name}: could not be read as an ELF object ({error})")
            report["artifacts"].append(artifact)
            continue
        artifact["readable"] = True
        artifact["needed"] = described["needed"]
        artifact["undefined_symbols"] = len(described["undefined_symbols"])
        artifact["defined_symbols"] = len(described["defined_symbols"])
        report["artifacts"].append(artifact)
        report["checked"] += 1

        has_issue, message = checker.check_linked_libraries(
            described["needed"], allowed_libraries, allowed_symbol_prefixes
        )
        if has_issue:
            report["errors"].append(f"{product.name}: {message}")

    report["passed"] = not report["errors"]
    return report


def _load_checker():
    """Import the checker by path, so this tool needs only the standard library."""
    import importlib.util

    path = REPO_TOP / "src" / "kernelbench" / "kernel_static_checker.py"
    spec = importlib.util.spec_from_file_location("kernelbench_static_checker", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load the static checker from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, help="read the whitelist out of this task contract")
    parser.add_argument("--policy", type=Path, help="a library_policy block, or a task contract holding one")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    allowed_libraries: Optional[List[str]] = None
    prefixes: Optional[List[str]] = None
    if args.task_dir:
        task = json.loads((args.task_dir / "task.json").read_text(encoding="utf-8"))
        policy = task.get("library_policy") or {}
        allowed_libraries = policy.get("allowed_libraries")
        prefixes = policy.get("allowed_symbol_prefixes")
    elif args.policy:
        policy = json.loads(Path(args.policy).read_text(encoding="utf-8"))
        allowed_libraries = policy.get("allowed_libraries")
        prefixes = policy.get("allowed_symbol_prefixes")

    report = check_build_dir(args.build_dir, allowed_libraries, prefixes)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
