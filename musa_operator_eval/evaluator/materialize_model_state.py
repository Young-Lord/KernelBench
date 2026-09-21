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
        # numpy has no bfloat16, so the same bits are read as a 16-bit integer and written
        # back unchanged. The view has to name a dtype this torch actually has: `uint16`
        # is not one of them, and asking for it raised AttributeError, so a bfloat16 case's
        # state was never materialized at all -- it failed loudly, which is why no bf16
        # compiled run had ever got past staging.
        return tensor.view(torch.int16).numpy().tobytes()
    return tensor.numpy().tobytes()


#: The two vocabularies this argument arrives in: the drivers' short one (`fp16`) and the
#: one the cases and manifests use (`bfloat16`). A caller that has a case in hand should
#: pass the case's own dtype, and the tool reads either spelling rather than making every
#: caller translate.
PRECISION_ALIASES = {
    "fp16": "fp16", "float16": "fp16", "half": "fp16",
    "bf16": "bf16", "bfloat16": "bf16",
    "fp32": "fp32", "float32": "fp32", "float": "fp32",
}


def normalize_precision(value: str) -> str:
    """The driver's spelling of a precision the case may have named in torch's."""
    try:
        return PRECISION_ALIASES[str(value).lower()]
    except KeyError:
        raise argparse.ArgumentTypeError(
            f"{value!r} is not a precision; expected one of {sorted(set(PRECISION_ALIASES))}"
        )


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


def build_state(task_dir: Path, case: dict, task: dict, problem_source: str, dtype) -> Dict[str, object]:
    """The reference's state, built the way the framework builds it -- for THIS case.

    The case re-binds module constants (`case_parameters` maps a case field onto the
    name a problem reads), so the model a case's golden came from is the one built
    from the *specialized* source, not from `problem.py` as written. Building the
    raw file here was wrong for every case whose configuration differs from the
    problem's defaults: `kb_l3_43_b`'s cases re-bind `n_embd`/`n_head`/`max_seqlen`,
    and `nn.Linear`'s weight shapes follow `n_embd`, so eight of its ten cases were
    handed a state of the wrong shape -- and a compiled submission that read it would
    have been graded as a wrong kernel rather than as a staging failure.

    Seeded as `kernelbench.eval` seeds it and cast to the case's dtype, because that
    is the state the graded model holds -- the reference and the submission are both
    cast before they run.
    """
    import torch

    sys.path.insert(0, str(REPO_TOP / "src"))
    sys.path.insert(0, str(REPO_TOP / "musa_operator_eval"))
    from kernelbench.eval import set_seed
    from tools.case_specialization import case_source

    # One more place that has to specialize a case exactly the way the grader does:
    # if this disagreed with `run_task.py` or `generate_cases.py`, the handed-over
    # state would describe a case nobody runs, and it would still look like a model.
    namespace: dict = {}
    exec(case_source(problem_source, case, task.get("case_parameters") or {}), namespace)

    set_seed(int(case["seed"]))
    init_inputs = namespace["get_init_inputs"]()
    set_seed(int(case["seed"]))
    with torch.no_grad():
        model = namespace["Model"](*init_inputs)
        model = model.to(dtype=dtype)
    return {key: value for key, value in model.state_dict().items()}, list(init_inputs)


def materialize(task_dir: Path, case: dict, task: dict, problem_source: str, dtype, output_dir: Path) -> dict:
    """Write the state to `output_dir` and describe it."""
    output_dir.mkdir(parents=True, exist_ok=True)
    state, init_inputs = build_state(task_dir, case, task, problem_source, dtype)
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
        "seed": int(case["seed"]),
        "case_id": case["case_id"],
        # Which configuration the state was built from, so a reader can see that the
        # case's parameters were used rather than the problem file's defaults.
        "init_inputs": [_plain(value) for value in init_inputs],
        "dtype": dtype_name(dtype),
    }


def _plain(value):
    """The init arguments as JSON: `get_init_inputs` returns numbers, and nothing else here."""
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    raise SystemExit(f"a case's init argument is a {type(value).__name__}, which has no place in a manifest")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize a reference's state as case inputs")
    parser.add_argument("--task-dir", required=True, help="the task package whose problem.py builds the model")
    parser.add_argument("--cases", type=Path, required=True, help="the case manifest the case comes from")
    parser.add_argument("--case", required=True, help="the case id to build the state for")
    parser.add_argument("--precision", default="fp16", type=normalize_precision,
                        help="the case's dtype; fp16/bf16/fp32 and float16/bfloat16/float32 are both accepted")
    parser.add_argument("--output-dir", required=True, help="where the state's .bin files go")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    sys.path.insert(0, str(REPO_TOP / "src"))
    sys.path.insert(0, str(REPO_TOP / "musa_operator_eval"))
    from kernelbench.eval import get_torch_dtype_from_string
    from tools.case_specialization import load_task

    dtype = get_torch_dtype_from_string(args.precision)
    # The case does not travel as a bare seed: the construction depends on the case's
    # parameters as well, so the manifest it came from has to be the one it is built
    # from. `--case` is required for the same reason -- building some case's state
    # from the problem's defaults is what this tool used to do, silently.
    task, cases_manifest, problem_source = load_task(Path(args.task_dir), args.cases)
    case = None
    for entry in cases_manifest.get("cases", []):
        if entry.get("case_id") == args.case:
            case = entry
            break
    if case is None:
        raise SystemExit(f"the manifest declares no case {args.case!r}")
    result = materialize(Path(args.task_dir), case, task, problem_source, dtype, Path(args.output_dir))
    sys.stdout.write(json.dumps(result) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
