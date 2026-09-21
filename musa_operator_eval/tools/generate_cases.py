"""Generate a case's inputs and golden tensors from the task package that owns it.

§4.3 keeps a case list to parameters and seeds: the tensors are produced from the
seed and the golden is stored beside them rather than inside the manifest. This
tool is what turns one case entry into that pair.

Everything it needs is declared by the task package, so it works for every family
rather than for the one whose shape vocabulary happened to be written into this
file first:

  * the inputs and the model come from the package's `problem.py`, specialised to
    the case and seeded from `case.seed`, cast to `case.dtype`;
  * the golden comes from the package's `golden_oracle` when it declares one --
    §5 step 3's separate CPU implementation -- and otherwise from the package's
    own reference model evaluated in float64 on the CPU.

Determinism is the point. The same case has to produce the same bytes wherever it
runs, or `case.seed` stops meaning anything and the evaluator's golden cross-check
turns into a coin flip. So the seed is applied the way `kernelbench.eval` applies
it (before `get_init_inputs`, again before the model is built, again before
`get_inputs`), and the reference is put in `eval()` mode: a graded case sets both
dropout rates to zero, and `eval()` is what makes that a fact about the run rather
than a fact about the case.

Writing into a task package is refused. A hidden golden inside `tasks/` would be
readable by the agent, which is the separation §2 of the guide is built on, and
the failure would be invisible -- the file would simply be there.

    # a task package's public smoke cases, before submitting
    python musa_operator_eval/tools/generate_cases.py --task-dir musa_operator_eval/tasks/kb_l3_43_b

    # the hidden set, from the evaluator side
    python musa_operator_eval/tools/generate_cases.py \
        --task-dir musa_operator_eval/tasks/kb_l3_43_b \
        --manifest musa_operator_eval/private/mingpt_causal_attention_b_v0/cases.private.json \
        --output-dir musa_operator_eval/private/generated
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from case_specialization import case_source, load_task  # noqa: E402

DEFAULT_TASK_DIR = ROOT / "tasks" / "sdpa_forward_pilot"
DEFAULT_GENERATED_SUBDIR = "generated"

# The file name a case's tensor manifest is written under, in both directions.
# §4.6 freezes the ABI: a directory of raw `.bin` files plus this manifest, and
# `run_task.py` resolves a case's `golden` path against it.
TENSORS_MANIFEST = "tensors.json"

# The precision the reference is evaluated at. §5 step 3 asks for an independent
# CPU fp64 reference; float64 on the CPU is what makes a golden a statement about
# the contract rather than about one build's fast-math settings.
REFERENCE_DTYPE = "float64"

# What a golden manifest records about which of the two paths produced it: the
# package's declared oracle, or the package's own model.
ORACLE_REFERENCE = "golden_oracle"
MODEL_REFERENCE = "cpu_fp64_reference"

# What a tensor's `layout` field says when the contract names one layout for the
# whole task and nothing more specific. A task that declares none gets the honest
# answer instead of a guess.
DEFAULT_LAYOUT = "contiguous"


# ---------------------------------------------------------------------------
# Storage format
# ---------------------------------------------------------------------------


def float32_to_bfloat16(values) -> "Any":
    import numpy as np

    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    rounding = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return ((bits + rounding) >> np.uint32(16)).astype("<u2")


def bfloat16_to_float32(values) -> "Any":
    import numpy as np

    return (np.asarray(values, dtype=np.uint16).astype(np.uint32) << np.uint32(16)).view(np.float32)


def quantize(values, dtype: str) -> Tuple["Any", "Any"]:
    """Round float32 values to a storage dtype; return (stored, as_float32).

    The second value is what the reference sees. Encoding a tensor and then
    computing from the unrounded numbers would produce a golden for inputs that
    were never written to disk.
    """
    import numpy as np

    if dtype == "float16":
        stored = np.asarray(values, dtype="<f2")
        return stored, stored.astype(np.float32)
    if dtype == "float32":
        stored = np.asarray(values, dtype="<f4")
        return stored, stored
    if dtype == "bfloat16":
        stored = float32_to_bfloat16(values)
        return stored, bfloat16_to_float32(stored)
    raise ValueError(f"unsupported dtype {dtype}")


def tensor_record(name: str, filename: str, dtype: str, shape: Sequence[int], data: bytes, layout: str) -> dict:
    return {
        "name": name,
        "file": filename,
        "dtype": dtype,
        "shape": [int(dimension) for dimension in shape],
        "layout": layout,
        "byte_order": "little",
        "nbytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def write_tensor(directory: Path, name: str, values, dtype: str, layout: str) -> Tuple[dict, "Any"]:
    """Write one tensor as a raw `.bin` file and return its manifest record."""
    stored, actual = quantize(values, dtype)
    data = stored.tobytes(order="C")
    filename = f"{name}.bin"
    (directory / filename).write_bytes(data)
    return tensor_record(name, filename, dtype, values.shape, data, layout), actual.reshape(values.shape)


def write_manifest(directory: Path, case_id: str, records: List[dict], reference: str) -> None:
    manifest = {
        "schema_version": "1.0.0",
        "case_id": case_id,
        "reference": reference,
        "tensors": records,
    }
    (directory / TENSORS_MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# The problem under one case
# ---------------------------------------------------------------------------


def problem_namespace(task: dict, problem_source: str, case: dict) -> dict:
    """Execute the problem source specialised to one case.

    An empty namespace is deliberate: every problem imports what it uses, so the
    context a caller supplies would only be a second place the same names are
    bound.
    """
    namespace: dict = {}
    exec(case_source(problem_source, case, task.get("case_parameters") or {}), namespace)
    return namespace


def set_seed(seed: int) -> None:
    """Seed the CPU generator the way the framework seeds every construction."""
    import torch

    torch.manual_seed(seed)


def case_inputs(namespace: dict, case: dict) -> List["Any"]:
    """Build the case's inputs, seeded and rounded to the case's storage dtype.

    Returned as float32 NumPy arrays: that is the precision the `.bin` files hold
    after decoding, and computing from anything else would describe inputs that
    were never written.
    """
    import torch

    set_seed(case["seed"])
    raw = namespace["get_inputs"]()
    tensors = []
    for value in raw:
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"case {case['case_id']}: get_inputs returned a {type(value).__name__}; "
                "the frozen ABI transfers tensors, so a non-tensor input has no file to go in"
            )
        tensors.append(value.detach().to(dtype=torch.float32).numpy())
    return tensors


def output_names(count: int) -> List[str]:
    """The golden's tensor names, which the ABI keeps positional and stable."""
    if count == 1:
        return ["output"]
    return [f"output_{index}" for index in range(count)]


def reference_golden(
    task: dict, task_dir: Path, namespace: dict, case: dict, inputs: List["Any"]
) -> Tuple[List[str], List["Any"], str]:
    """Compute the golden for one case.

    Returns:
        (names, values, description of what produced them)

    The oracle path exists because §5 step 3 asks for a reference separate from the
    package's own model. Where a package ships one, it is the oracle: a second
    implementation disagreeing with the first is the only thing that can find a
    reference that is wrong rather than merely different from the submission.

    An oracle is named relative to its own package, because that is where §3 puts
    it and because a package has to stay movable.
    """
    import numpy as np
    import torch

    oracle = task.get("golden_oracle")
    if oracle:
        path = Path(oracle["path"])
        if not path.is_absolute():
            path = Path(task_dir) / path
        spec = importlib.util.spec_from_file_location("golden_oracle", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load the declared golden oracle from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        entrypoint = getattr(module, oracle["entrypoint"])
        produced = entrypoint(*inputs, **(case.get("attributes") or {}))
        values = list(produced) if isinstance(produced, (tuple, list)) else [produced]
        names = list(oracle["returns"])
        if len(names) != len(values):
            raise ValueError(
                f"the declared golden oracle returns {len(values)} tensors but {len(names)} are declared"
            )
        return names, [np.asarray(value) for value in values], ORACLE_REFERENCE

    set_seed(case["seed"])
    init_inputs = namespace["get_init_inputs"]()
    with torch.no_grad():
        set_seed(case["seed"])
        model = namespace["Model"](*init_inputs)
        model = model.eval().to(dtype=getattr(torch, REFERENCE_DTYPE))
        doubled = [torch.from_numpy(np.asarray(value, dtype="float64")) for value in inputs]
        produced = model(*doubled)

    if isinstance(produced, (tuple, list)):
        values = list(produced)
    else:
        values = [produced]
    return output_names(len(values)), [value.detach().numpy() for value in values], MODEL_REFERENCE


# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------


def golden_dtype_for(index: int, case: dict) -> str:
    """The storage dtype of one golden tensor.

    The first tensor is the model's output and is stored at the case's dtype, so
    the golden is comparable to what a submission returns. A companion tensor --
    the pilot's log-sum-exp is the one that exists -- is a property of the oracle
    rather than of the model and is stored at float32.
    """
    return case["dtype"] if index == 0 else "float32"


def generate_case(case: dict, output_root: Path, task_dir: Optional[Path] = None) -> Path:
    """Write one case's inputs and golden under `output_root`.

    Args:
        case: a §4.3 case entry
        output_root: the directory the case's `<case_id>/{input,golden}/` goes in
        task_dir: the task package the case belongs to. Defaults to the pilot,
            which is the one package whose cases were generated before this tool
            read `task.json`.

    Returns:
        The case's directory under `output_root`.
    """
    task_dir = Path(task_dir) if task_dir else DEFAULT_TASK_DIR
    task, _cases, problem_source = load_task(task_dir)
    namespace = problem_namespace(task, problem_source, case)

    case_root = Path(output_root) / case["case_id"]
    input_dir, golden_dir = case_root / "input", case_root / "golden"
    input_dir.mkdir(parents=True, exist_ok=True)
    golden_dir.mkdir(parents=True, exist_ok=True)

    layouts = (task.get("tensor_contract") or {}).get("layouts") or []
    layout = layouts[0] if len(layouts) == 1 else DEFAULT_LAYOUT

    inputs = case_inputs(namespace, case)
    input_names = (task.get("tensor_contract") or {}).get("input_names") or [
        f"input_{index}" for index in range(len(inputs))
    ]
    if len(input_names) != len(inputs):
        raise ValueError(
            f"{task.get('id')} declares {len(input_names)} input names for {len(inputs)} inputs"
        )

    records = []
    rounded = []
    for name, values in zip(input_names, inputs):
        record, actual = write_tensor(input_dir, name, values, case["dtype"], layout)
        records.append(record)
        rounded.append(actual)
    write_manifest(input_dir, case["case_id"], records, "get_inputs")

    names, values, description = reference_golden(task, task_dir, namespace, case, rounded)
    golden_records = []
    for index, (name, tensor) in enumerate(zip(names, values)):
        dtype = golden_dtype_for(index, case)
        record, _stored = write_tensor(golden_dir, name, tensor, dtype, layout)
        golden_records.append(record)
    write_manifest(golden_dir, case["case_id"], golden_records, description)

    return case_root


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def output_is_inside_a_task_package(output_dir: Path) -> bool:
    """Whether writing there would put a tensor in the agent-visible tree."""
    resolved = Path(output_dir).resolve()
    tasks_root = (ROOT / "tasks").resolve()
    return resolved == tasks_root or tasks_root in resolved.parents


def generate_manifest(manifest: dict, output_root: Path, task_dir: Path) -> List[str]:
    """Generate every case in a manifest and return the case ids written."""
    written = []
    for case in manifest["cases"]:
        generate_case(case, output_root, task_dir)
        written.append(case["case_id"])
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--task-dir", type=Path, default=DEFAULT_TASK_DIR)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="case manifest to generate; defaults to the task's public_cases.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="where <case_id>/{input,golden}/ goes; defaults to <task-dir>/generated",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    task_dir = args.task_dir
    manifest_path = args.manifest or (task_dir / "public_cases.json")
    output_dir = args.output_dir or (task_dir / DEFAULT_GENERATED_SUBDIR)

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))

    # A public case's tensors are meant to be readable: they are the smoke cases an
    # agent self-tests against, and writing them beside their manifest is the one
    # place that needs no explaining. A hidden case's are not, and the failure
    # would be invisible -- the file would simply sit in the tree the agent reads.
    if manifest.get("visibility") == "private" and output_is_inside_a_task_package(output_dir):
        print(
            f"refusing to write hidden tensors into {output_dir}: a task package is the "
            "agent-visible tree, so a hidden golden inside it is a published golden. "
            "Point --output-dir outside musa_operator_eval/tasks/.",
            file=sys.stderr,
        )
        return 2

    for case_id in generate_manifest(manifest, Path(output_dir), Path(task_dir)):
        print(f"generated {case_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
