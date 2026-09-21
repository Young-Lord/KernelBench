"""Materialize a task's reference model state as tensors a submission can read.

§4.6's compiled path grades a case out of what its input directory holds, and for a
family whose reference builds weights in `__init__` those weights are part of the
model rather than of `get_inputs()`. A torch-free submission cannot reconstruct them:
they are a function of the case's seed and of the construction order, and §4.3 keeps a
case to parameters and seeds rather than tensors. So the evaluator, which does have
torch, materializes them and hands them over the same way it hands over the case's own
tensors.

The construction is the framework's, in the framework's order (`kernelbench.eval`):
seed, `get_init_inputs()`, seed again, `Model(*init_inputs)`, cast to the case's dtype.
Run that way the state is the state the graded model holds, which is why no golden has
to be regenerated for this: the golden was produced from the same seed stream.

This tool writes the `.bin` files and reports what it wrote. It does not touch the
generated tree -- the caller stages a copy of the case's input directory and merges
what this prints into the staged manifest, so a case's own inputs stay what they were.

Run with:
    python musa_operator_eval/evaluator/materialize_model_state.py \
        --task-dir musa_operator_eval/tasks/kb_l3_43_b \
        --case hidden_perf_001 --seed 91007 --precision fp16 --output-dir /tmp/state
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_TOP = Path(__file__).resolve().parents[2]


def state_tensor_name(key: str) -> str:
    """The name a submission sees for one entry of the reference's state.

    The state dict's own key, prefixed so that it cannot collide with a case's
    inputs (`input_0`, or the pilot's `lse`), and kept verbatim otherwise: `c_attn.weight`
    arrives as `state_c_attn.weight`, which is the name the module gave it.
    """
    return f"state_{key}"


def state_file_name(key: str) -> str:
    return f"{state_tensor_name(key)}.bin"


def state_digest(records: List[dict]) -> str:
    """One digest over the whole state, so a drift is a named mismatch and not a wrong number.

    Ordered by tensor name, so a differently ordered state_dict is still the same
    state, and built from the per-tensor digests, so a changed value anywhere changes
    the whole thing.
    """
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item["name"]):
        digest.update(f"{record['name']}:{record['sha256']}\n".encode("utf-8"))
    return digest.hexdigest()


def raw_bytes(tensor) -> bytes:
    """A tensor's storage, little-endian, without numpy's gaps on the two-byte floats."""
    import torch

    tensor = tensor.detach().cpu().contiguous()
    if tensor.dtype == torch.bfloat16:
        # numpy has no bfloat16; the same bits read as uint16 keep the storage exact.
        return tensor.view(torch.uint16).numpy().tobytes()
    return tensor.numpy().tobytes()


def dtype_name(dtype) -> str:
    """The manifest's spelling of a torch dtype, matching what the reader expects."""
    import torch

    if dtype == torch.float32:
        return "float32"
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.bfloat16:
        return "bfloat16"
    raise ValueError(f"the state holds {dtype}, which the tensor manifest cannot spell")


def build_state(task_dir: Path, seed: int, dtype) -> Dict[str, object]:
    """The reference's state, built the way the framework builds it.

    Seeded as `kernelbench.eval` seeds it and cast to the case's dtype, because that
    is the state the graded model holds -- the reference and the submission are both
    cast before they run.
    """
    import importlib.util

    import torch

    sys.path.insert(0, str(REPO_TOP / "src"))
    from kernelbench.eval import set_seed

    spec = importlib.util.spec_from_file_location("materialize_problem", task_dir / "problem.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    set_seed(int(seed))
    init_inputs = module.get_init_inputs()
    set_seed(int(seed))
    with torch.no_grad():
        model = module.Model(*init_inputs)
        model = model.to(dtype=dtype)
    return {key: value for key, value in model.state_dict().items()}


def materialize(task_dir: Path, seed: int, dtype, output_dir: Path) -> dict:
    """Write the state to `output_dir` and describe it."""
    output_dir.mkdir(parents=True, exist_ok=True)
    state = build_state(task_dir, seed, dtype)
    records: List[dict] = []
    for key in sorted(state):
        payload = raw_bytes(state[key])
        path = output_dir / state_file_name(key)
        path.write_bytes(payload)
        records.append(
            {
                "name": state_tensor_name(key),
                "file": state_file_name(key),
                "dtype": dtype_name(state[key].dtype),
                "shape": [int(axis) for axis in state[key].shape],
                "layout": "contiguous",
                "byte_order": "little",
                "nbytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return {
        "state_dict_keys": len(state),
        "tensors": records,
        "digest": state_digest(records),
        "seed": int(seed),
        "dtype": dtype_name(dtype),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize a reference's state as case inputs")
    parser.add_argument("--task-dir", required=True, help="the task package whose problem.py builds the model")
    parser.add_argument("--seed", required=True, type=int, help="the case's seed")
    parser.add_argument("--precision", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--output-dir", required=True, help="where the state's .bin files go")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    sys.path.insert(0, str(REPO_TOP / "src"))
    from kernelbench.eval import get_torch_dtype_from_string

    dtype = get_torch_dtype_from_string(args.precision)
    result = materialize(Path(args.task_dir), args.seed, dtype, Path(args.output_dir))
    sys.stdout.write(json.dumps(result) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
