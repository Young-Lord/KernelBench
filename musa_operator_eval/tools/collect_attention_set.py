"""Verify the MUSA attention set against the repository and regenerate its index.

The attention set is a curation decision, so `attention_set.json` is hand-written
and checked in. Everything else about it is a claim about files in this
repository, and this tool is what keeps those claims honest:

- every reference and A-tier answer path must exist, and its sha256 is recorded;
- every measurement must still match the file it was read from, so a re-run
  cannot silently leave a stale number in the index;
- the operator inventory is extracted from the references rather than
  transcribed, so it cannot rot;
- the dangling provenance references are re-checked and reported, not assumed.

`SET.md` is generated, never edited by hand. Exit code is non-zero on drift, so
this can gate a commit.

    python musa_operator_eval/tools/collect_attention_set.py            # verify + write
    python musa_operator_eval/tools/collect_attention_set.py --check    # verify only
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
REPO_TOP = ROOT.parent

SET_DIR = ROOT / "sets" / "attention"
MANIFEST = SET_DIR / "attention_set.json"
INDEX = SET_DIR / "SET.md"

ATTN_PERF_CSV = "runs/musa_attn_perf_results.csv"
LEVEL1_BASELINE = "deepseek_api_level1_musa_s4000_baseline"

# Recorded measurements are rounded for readability, so comparison is
# approximate. Tighter than this and every legitimate re-rounding trips it.
RELATIVE_TOLERANCE = 2e-3

# Tensor-level operations worth naming in the inventory. Scope methods to this
# list rather than matching every `.foo(` call, which would be noise.
TENSOR_METHODS = (
    "matmul", "mm", "bmm", "softmax", "log_softmax", "relu", "gelu", "silu",
    "masked_fill", "transpose", "permute", "view", "reshape", "contiguous",
    "expand", "repeat", "repeat_interleave", "unsqueeze", "squeeze", "amax",
    "sum", "mean", "max", "min", "clamp", "mul", "add", "div", "pow", "tanh",
    "split", "flatten", "unflatten", "dropout", "layer_norm", "drop_path",
)


class DriftError(Exception):
    """A recorded claim no longer matches the repository."""


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> List[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


# ---------------------------------------------------------------------------
# Measurement readers. Each returns millisecond floats so the recorded values
# can be compared directly; unit conversion belongs here, not in the manifest.
# ---------------------------------------------------------------------------

def read_attn_perf_csv(selector: dict) -> dict:
    """One row of runs/musa_attn_perf_results.csv, keyed by problem and timestamp.

    The file accumulates re-runs, so the timestamp is what picks a version out:
    a problem appears once per tuning round.
    """
    path = REPO_TOP / ATTN_PERF_CSV
    matches = [
        row for row in _rows(path)
        if row["problem"] == str(selector["problem"]) and row["ts"] == selector["ts"]
    ]
    if not matches:
        raise DriftError(f"{ATTN_PERF_CSV}: no row for problem {selector['problem']} at {selector['ts']}")
    row = matches[0]
    return {
        "reference_ms": float(row["ref_ms"]),
        "submission_ms": float(row["ours_ms"]),
        "speedup": float(row["speedup"]),
    }


def read_kernelbench_eval_json(selector: dict) -> dict:
    """One sample of deepseek_api_level1_musa_s4000_baseline/eval_results.json.

    `runtime` is already in milliseconds; the archive's README states the timing
    unit, and the speedups in analysis/pytorch_vs_deepseek_musa only reconcile
    under that reading.
    """
    path = REPO_TOP / LEVEL1_BASELINE / "eval_results.json"
    archive = json.loads(path.read_text(encoding="utf-8"))
    samples = archive.get(str(selector["problem"]))
    if not samples:
        raise DriftError(f"eval_results.json: no entry for problem {selector['problem']}")
    sample = samples[selector.get("sample_id", 0)]
    return {
        "submission_ms": float(sample["runtime"]),
        "correctness": bool(sample["correctness"]),
        "compiled": bool(sample["compiled"]),
    }


READERS = {
    "attn_perf_csv": read_attn_perf_csv,
    "kernelbench_eval_json": read_kernelbench_eval_json,
}


def _close(recorded: float, actual: float) -> bool:
    if recorded == actual:
        return True
    scale = max(abs(recorded), abs(actual), 1e-12)
    return abs(recorded - actual) / scale <= RELATIVE_TOLERANCE


# ---------------------------------------------------------------------------
# Reference inspection
# ---------------------------------------------------------------------------

def extract_operators(source: str) -> Dict[str, List[str]]:
    """Names of the torch-level operations a reference implementation calls.

    A regex scan, deliberately: the point is an inventory a reader can trust to
    be derived from the source rather than transcribed, not a call graph.
    """
    found = {
        "torch": sorted(set(re.findall(r"\btorch\.([a-z_][\w]*(?:\.[a-z_]\w*)*)\s*\(", source))),
        "functional": sorted(set(re.findall(r"\bF\.([a-z_]\w*)\s*\(", source))),
        "modules": sorted(set(re.findall(r"\bnn\.([A-Z]\w*)\s*\(", source))),
        "tensor_methods": sorted({
            name for name in TENSOR_METHODS
            if re.search(rf"\.{re.escape(name)}\s*\(", source)
        }),
    }
    return {key: value for key, value in found.items() if value}


def verify_entry(entry: dict) -> Tuple[dict, List[str]]:
    """Check one entry's file and measurement claims. Returns (facts, problems)."""
    problems: List[str] = []
    facts: Dict[str, Any] = {"entry_id": entry["entry_id"]}

    reference_path = REPO_TOP / entry["reference"]["path"]
    if not reference_path.is_file():
        problems.append(f"{entry['entry_id']}: reference missing at {entry['reference']['path']}")
        return facts, problems
    reference_source = reference_path.read_text(encoding="utf-8")
    facts["reference"] = {
        "path": entry["reference"]["path"],
        "sha256": sha256_of(reference_path),
        "lines": len(reference_source.splitlines()),
    }
    facts["operators"] = extract_operators(reference_source)

    answer = entry.get("tier", {}).get("A_kernel", {}).get("answer")
    if answer:
        answer_path = REPO_TOP / answer
        if not answer_path.is_file():
            problems.append(f"{entry['entry_id']}: A-tier answer missing at {answer}")
        else:
            facts["answer"] = {
                "path": answer,
                "sha256": sha256_of(answer_path),
                "lines": len(answer_path.read_text(encoding="utf-8").splitlines()),
            }

    measurements = []
    for measurement in entry.get("measurements", []):
        reader = READERS.get(measurement["reader"])
        if reader is None:
            problems.append(f"{entry['entry_id']}: unknown reader {measurement['reader']!r}")
            continue
        try:
            actual = reader(measurement["selector"])
        except DriftError as error:
            problems.append(f"{entry['entry_id']}/{measurement['measurement_id']}: {error}")
            continue

        for field, recorded in measurement["expected"].items():
            if field not in actual:
                problems.append(
                    f"{entry['entry_id']}/{measurement['measurement_id']}: "
                    f"reader produced no {field!r}"
                )
                continue
            if isinstance(recorded, bool) or isinstance(actual[field], bool):
                if recorded != actual[field]:
                    problems.append(
                        f"{entry['entry_id']}/{measurement['measurement_id']}: "
                        f"{field} recorded {recorded} but source says {actual[field]}"
                    )
            elif not _close(float(recorded), float(actual[field])):
                problems.append(
                    f"{entry['entry_id']}/{measurement['measurement_id']}: "
                    f"{field} recorded {recorded} but source says {actual[field]}"
                )

        measurements.append({
            "measurement_id": measurement["measurement_id"],
            "role": measurement["role"],
            "target_id": measurement["target_id"],
            "source": measurement["reader"],
            "actual": actual,
        })
    facts["measurements"] = measurements
    return facts, problems


def verify_dangling(entries: List[dict]) -> List[dict]:
    """Re-check every source path the set records.

    Two kinds live here and they are not the same defect. A path with
    `resolves_on` is expected to be absent from this working tree but present in
    that branch, which is where the vendored upstream archive lives. A path
    without it is expected to be absent everywhere, and the check is simply that
    it stayed that way.
    """
    results = []
    for reference in entries:
        path = REPO_TOP / reference["path"]
        exists_locally = path.exists()
        record = {**reference, "exists_locally": exists_locally}
        if reference.get("resolves_on"):
            record["branch_reachable"] = _path_exists_in_ref(reference["path"], reference["resolves_on"])
        results.append(record)
    return results


def _path_exists_in_ref(path: str, branch: str) -> Optional[bool]:
    """Whether `path` exists in `branch`, or None when the branch is unreachable.

    Three ref spellings are tried, because how the archive is reachable depends
    on how it was fetched: a single-branch clone has no `origin/<branch>`, and a
    fresh `git fetch origin <branch>` leaves only FETCH_HEAD. None rather than
    False on an unreachable branch, because never having fetched the archive is a
    different situation from the archive being gone.

    The two failure messages are distinct (`invalid object name` for a ref that
    does not resolve, `path ... does not exist` for a path inside a ref that
    does), which is what makes the distinction reliable.
    """
    import subprocess

    for ref in (branch, f"origin/{branch}", "FETCH_HEAD"):
        try:
            completed = subprocess.run(
                ["git", "cat-file", "-e", f"{ref}:{path}"],
                cwd=REPO_TOP, capture_output=True, timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if completed.returncode == 0:
            return True
        if b"invalid object name" not in completed.stderr:
            # The ref resolved and the path inside it did not. No other spelling
            # will do better.
            return False
    return None


# ---------------------------------------------------------------------------
# Kernel candidates
# ---------------------------------------------------------------------------

CANDIDATE_KINDS = ("fused_library", "expert_dispatch_source", "handwritten_kernel")

REQUIRED_CANDIDATE_FIELDS = ("candidate_id", "name", "kind", "targets", "why_it_matters")


def load_candidate_record(manifest: dict) -> dict:
    """Load the kernel candidate record the set points at."""
    return json.loads((REPO_TOP / manifest["kernel_candidates_record"]).read_text(encoding="utf-8"))


def _has_evidence(candidate: dict) -> bool:
    """Whether a candidate points at something a reader can go and check.

    Archive-derived candidates carry `archive_path` into the source archive;
    in-repo ones carry the same field pointing at a repository path. A bare
    `evidence` list of upstream pages is accepted for anything that is neither.
    """
    return bool(candidate.get("archive_path") or candidate.get("evidence"))


def verify_candidates(manifest: dict, record: dict) -> Tuple[List[dict], List[str]]:
    """Check the candidate record and the entry-to-candidate references.

    The two files declare the same relationship from opposite ends: an entry
    lists `candidate_kernels`, a candidate lists `serves`. Checking only one
    direction would let the two drift, so both are checked and they must agree.
    """
    problems: List[str] = []
    candidates = record.get("candidates") or []
    if not candidates:
        return [], ["the kernel candidate record lists no candidates"]

    known_targets = {target["target_id"] for target in manifest["targets"]}
    known_ids = set()
    declared_serves: Dict[str, List[str]] = {}

    for candidate in candidates:
        candidate_id = candidate.get("candidate_id", "<unnamed>")
        for field in REQUIRED_CANDIDATE_FIELDS:
            if not candidate.get(field):
                problems.append(f"candidate {candidate_id}: missing {field!r}")
        if candidate_id in known_ids:
            problems.append(f"candidate {candidate_id}: duplicate id")
        known_ids.add(candidate_id)
        if candidate.get("kind") and candidate["kind"] not in CANDIDATE_KINDS:
            problems.append(f"candidate {candidate_id}: unknown kind {candidate['kind']!r}")
        for target in candidate.get("targets", []):
            if target not in known_targets:
                problems.append(f"candidate {candidate_id}: unknown target {target!r}")
        if not _has_evidence(candidate):
            problems.append(
                f"candidate {candidate_id}: a claim with no evidence is not checkable"
            )
        declared_serves[candidate_id] = list(candidate.get("serves") or [])

    known_entry_ids = {entry["entry_id"] for entry in manifest["entries"]}
    for candidate_id, entry_ids in declared_serves.items():
        for entry_id in entry_ids:
            if entry_id not in known_entry_ids:
                problems.append(f"candidate {candidate_id}: serves unknown entry {entry_id!r}")

    for entry in manifest["entries"]:
        declared = list(entry.get("candidate_kernels") or [])
        entry_id = entry["entry_id"]
        for candidate_id in declared:
            if candidate_id not in known_ids:
                problems.append(f"entry {entry_id}: unknown candidate kernel {candidate_id!r}")
        # The reverse view, so a one-sided edit is caught.
        for candidate_id, entry_ids in declared_serves.items():
            if (entry_id in entry_ids) != (candidate_id in declared):
                problems.append(
                    f"{entry_id} and candidate {candidate_id} disagree: "
                    f"entry says {candidate_id in declared}, candidate says {entry_id in entry_ids}"
                )

    return candidates, problems


# ---------------------------------------------------------------------------
# Level 1 A-tier archive: the numbers the B-tier gate needs
# ---------------------------------------------------------------------------

def summarize_level1_archive() -> dict:
    """Aggregate the archived Level 1 A-tier run.

    This is the asset the B tier actually needs: the archived submission is a
    hand-written kernel with no library dispatch, so its latency is the
    composition-side number that the `naive / expert >= 1.3` admission gate is
    defined against. The gap between the library path and the archived kernel is
    therefore the first estimate of what a dispatch decision is worth on a
    problem, and it is what ranks the candidates below.
    """
    baseline = json.loads((REPO_TOP / LEVEL1_BASELINE / "baseline_time_torch.json").read_text(encoding="utf-8"))
    archive = json.loads((REPO_TOP / LEVEL1_BASELINE / "eval_results.json").read_text(encoding="utf-8"))
    reference_times = baseline["level1"]

    rows = []
    excluded = []
    for filename, stats in reference_times.items():
        problem_id = int(filename.split("_")[0])
        samples = archive.get(str(problem_id))
        if not samples:
            excluded.append({"problem_id": problem_id, "reason": "no record in eval_results.json"})
            continue
        sample = samples[0]
        library_ms = float(stats["mean"])
        naive_ms = float(sample["runtime"])
        if naive_ms <= 0:
            # A negative runtime is the evaluator's marker for "no timing taken".
            # Dropping it silently would make the archive look cleaner than it is.
            excluded.append({
                "problem_id": problem_id,
                "reason": f"no timing (runtime={naive_ms}) because correctness did not pass",
            })
            continue
        rows.append({
            "problem_id": problem_id,
            "name": filename.split("_", 1)[1].removesuffix(".py"),
            "library_ms": library_ms,
            "naive_ms": naive_ms,
            "library_gap": naive_ms / library_ms,
            "correctness": bool(sample["correctness"]),
        })
    rows.sort(key=lambda row: row["problem_id"])

    correct = [row for row in rows if row["correctness"]]
    gaps = sorted(correct, key=lambda row: row["library_gap"], reverse=True)
    return {
        "archived": len(reference_times),
        "compared": len(rows),
        "correct": len(correct),
        "excluded": excluded,
        "geometric_mean_speedup": _geometric_mean([row["library_ms"] / row["naive_ms"] for row in correct]),
        "median_speedup": _median([row["library_ms"] / row["naive_ms"] for row in correct]),
        "rows": rows,
        "largest_library_gaps": gaps[:20],
    }


def _geometric_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(math.exp(sum(math.log(value) for value in values) / len(values)))


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return float((ordered[middle - 1] + ordered[middle]) / 2)


# ---------------------------------------------------------------------------
# Index generation
# ---------------------------------------------------------------------------

def _format_speedup(actual: dict) -> str:
    if "speedup" in actual:
        return f"{actual['speedup']:.4f}x"
    return "-"


def render_index(manifest: dict, facts: List[dict], dangling: List[dict], level1: dict,
                 candidates: List[dict], record: dict) -> str:
    facts_by_id = {entry["entry_id"]: entry for entry in facts}
    record_headline = record.get("headline")
    lines: List[str] = []

    lines.append(f"# {manifest['title']}")
    lines.append("")
    lines.append("<!-- Generated by musa_operator_eval/tools/collect_attention_set.py. Do not edit by hand. -->")
    lines.append("")
    lines.append(manifest["purpose"])
    lines.append("")

    lines.append("## 目标设备与 dtype 矩阵")
    lines.append("")
    lines.append("| target | device | MUSA arch | warp size | dtype | status |")
    lines.append("|---|---|---|---:|---|---|")
    for target in manifest["targets"]:
        lines.append(
            f"| `{target['target_id']}` | {target['device']} | `{target['musa_arch']}` | "
            f"{target['warp_size']} | {target['dtype']} | {target['status']} |"
        )
    lines.append("")
    lines.append(f"> {manifest['warp_size_note']}")
    lines.append("")

    lines.append("## 题目清单")
    lines.append("")
    lines.append("| entry | KernelBench | 形状 | 类型 | A 类 | B 类 | 最终 speedup |")
    lines.append("|---|---|---|---|---|---|---:|")
    for entry in manifest["entries"]:
        entry_facts = facts_by_id.get(entry["entry_id"], {})
        kb = entry["kernelbench"]
        shapes = ", ".join(f"{key}={value}" for key, value in entry["shapes"].items())
        a_tier = entry["tier"]["A_kernel"]
        b_tier = entry["tier"]["B_library"]
        a_mark = "有答案" if a_tier.get("eligible") else "不计分"
        b_mark = "可判分" if b_tier.get("eligible") else "不计分"
        tuned = [
            measurement for measurement in entry_facts.get("measurements", [])
            if measurement["role"] == "a_tier_tuned"
        ]
        speedup = _format_speedup(tuned[0]["actual"]) if tuned else "-"
        lines.append(
            f"| `{entry['entry_id']}` | L{kb['level']}/{kb['problem_id']} {kb['name']} | {shapes} | "
            f"`{entry['attention_kind']}` | {a_mark} | {b_mark} | {speedup} |"
        )
    lines.append("")

    lines.append("## 可用 kernel 候选")
    lines.append("")
    lines.append(
        "可用于出题、做 expert dispatch 或作为参考实现的 attention kernel，一条一个实现。"
        "上游位置、版本与依赖写在溯源记录里"
        f"（`{manifest['kernel_candidates_record']}`，maintainer-only），这里只列选型需要的信息。"
    )
    lines.append("")
    lines.append("| candidate | 类型 | 覆盖 target | dtype | 服务题目 |")
    lines.append("|---|---|---|---|---|")
    entries_by_candidate: Dict[str, List[str]] = {}
    for entry in manifest["entries"]:
        for candidate_id in entry.get("candidate_kernels", []):
            entries_by_candidate.setdefault(candidate_id, []).append(entry["entry_id"])
    for candidate in candidates:
        targets = ", ".join(f"`{target}`" for target in candidate.get("targets", []))
        dtypes = ", ".join(candidate.get("dtypes", [])) or "未记录"
        serves = ", ".join(f"`{entry_id}`" for entry_id in entries_by_candidate.get(candidate["candidate_id"], [])) or "—"
        lines.append(
            f"| `{candidate['candidate_id']}` | `{candidate['kind']}` | {targets} | {dtypes} | {serves} |"
        )
    lines.append("")

    headline_candidate = candidates[0].get("targets") if candidates else None
    if record_headline:
        lines.append(f"> {record_headline}")
        lines.append("")

    for candidate in candidates:
        lines.append(f"### `{candidate['candidate_id']}` — {candidate['name']}")
        lines.append("")
        lines.append(f"- 类型：`{candidate['kind']}`")
        requirements = candidate.get("requirements") or {}
        if requirements:
            lines.append("- 运行要求：" + "；".join(f"{key} {value}" for key, value in requirements.items()))
        limits = candidate.get("limits") or {}
        if limits:
            lines.append("- 边界：" + "；".join(f"{key} — {value}" for key, value in limits.items()))
        lines.append(f"- {candidate['why_it_matters']}")
        lines.append("")

    record_gaps = record.get("gaps") or []
    if record_gaps:
        lines.append("候选清单记录的缺口：")
        lines.append("")
        for gap in record_gaps:
            lines.append(f"- {gap}")
        lines.append("")

    lines.append("## 逐题明细")
    lines.append("")
    for entry in manifest["entries"]:
        entry_facts = facts_by_id.get(entry["entry_id"], {})
        kb = entry["kernelbench"]
        lines.append(f"### {entry['entry_id']} — L{kb['level']}/{kb['problem_id']} {kb['name']}")
        lines.append("")
        lines.append(f"- 参考实现：`{entry['reference']['path']}`（`{entry['reference']['entrypoint']}`）")
        if entry_facts.get("reference"):
            lines.append(f"  - sha256 `{entry_facts['reference']['sha256'][:16]}…`，{entry_facts['reference']['lines']} 行")
        if entry_facts.get("answer"):
            lines.append(f"- A 类答案：`{entry_facts['answer']['path']}`（{entry_facts['answer']['lines']} 行）")
        lines.append(f"- 结构：{entry['structure']}")
        if entry.get("shape_note"):
            lines.append(f"- 形状说明：{entry['shape_note']}")
        lines.append("")
        lines.append("| 阶段 | target | 来源 | 参考 ms | 提交 ms | speedup |")
        lines.append("|---|---|---|---:|---:|---:|")
        for measurement in entry_facts.get("measurements", []):
            actual = measurement["actual"]
            lines.append(
                f"| {measurement['role']} | `{measurement['target_id']}` | `{measurement['source']}` | "
                f"{_format_number(actual.get('reference_ms'))} | "
                f"{_format_number(actual.get('submission_ms'))} | {_format_speedup(actual)} |"
            )
        lines.append("")
        for note in entry.get("notes", []):
            lines.append(f"- {note}")
        lines.append("")

    lines.append("## 参考实现用到的算子")
    lines.append("")
    lines.append("由本工具从参考源码正则提取，非手工转录。")
    lines.append("")
    for entry in manifest["entries"]:
        entry_facts = facts_by_id.get(entry["entry_id"], {})
        operators = entry_facts.get("operators") or {}
        if not operators:
            continue
        lines.append(f"- **{entry['entry_id']}**：" + "；".join(
            f"{kind} {', '.join(f'`{name}`' for name in names)}"
            for kind, names in operators.items()
        ))
    lines.append("")

    lines.append("## Level 1 A 类归档资产")
    lines.append("")
    lines.append(
        f"`{LEVEL1_BASELINE}/` 存档了 {level1['archived']} 道 Level 1 的手写 MUSA kernel 评测结果"
        f"（MTT S4000 / mp_22 / fp32，5 次正确性 + 100 次 MUSA event 计时）。"
        f"其中 {level1['compared']} 道进入了对照，"
        f"相对 torch 的几何均值 speedup {level1['geometric_mean_speedup']:.4f}，"
        f"中位数 {level1['median_speedup']:.4f}。"
    )
    lines.append("")
    if level1["excluded"]:
        lines.append("未进入对照的题：")
        lines.append("")
        for entry in level1["excluded"]:
            lines.append(f"- p{entry['problem_id']}：{entry['reason']}")
        lines.append("")
        lines.append(
            "> p72 的排除是评测侧的已知问题：该题的正确性对比是在与一个错位的"
            " muDNN grouped ConvTranspose3d 参考比较，归档里的 kernel 相对 CPU 标准公式逐位一致。"
            "它不应该被算成 A 类的失败，但在参考修好之前也不该进入 B 类的准入对照。"
        )
        lines.append("")
    lines.append(
        "这份归档同时是 B 类 `naive / expert >= 1.3` 准入用的 naive 侧数据："
        "归档里的提交是不做库分派的手写 kernel，与 B 类的兜底路径同量级。"
        "库路径与归档 kernel 的差距，就是该题上一次分派决策值多少钱的一阶估计。"
        "下面按这个差距排序。"
    )
    lines.append("")
    lines.append("| Level 1 | 题目 | torch 库 ms | 归档 kernel ms | 库领先 |")
    lines.append("|---:|---|---:|---:|---:|")
    for row in level1["largest_library_gaps"]:
        lines.append(
            f"| {row['problem_id']} | {row['name'][:52]} | {row['library_ms']:.3f} | "
            f"{row['naive_ms']:.3f} | {row['library_gap']:.1f}x |"
        )
    lines.append("")
    lines.append(
        "> 这些是 A 类题目编号，同时也是 B 类候选：库路径领先越多，"
        "「优先融合库、次选库组合、最后自研兜底」这个分派决策的分数差异越大。"
        "卷积族（55/56/62/63/67/80/87）和 matmul 族（1/2/3/11）是最集中的两片。"
    )
    lines.append("")

    lines.append("## 引用路径状态")
    lines.append("")
    lines.append(
        "这些路径被仓库内的文件引用为上游快照。要分清两种情况："
        f"标了分支的，本工作树里没有、但在 `{manifest['source_archive']['branch']}` 上存在——"
        "上游源码归档就放在那条分支，没有合进来，因为 `.gitignore` 已经确立了"
        "「vendored 上游 MUSA 源码按需取、不入库」的约定；"
        "没标分支的，是本机也确实不存在。"
    )
    lines.append("")
    lines.append("| 引用方 | 路径 | 角色 | 本工作树 | 所在分支 |")
    lines.append("|---|---|---|---|---|")
    for reference in dangling:
        branch = reference.get("resolves_on") or "—"
        if not reference["exists_locally"]:
            state = "缺失（预期）"
        else:
            state = "已就位"
        if reference.get("branch_reachable") is True:
            state += " / 分支可达"
        elif reference.get("branch_reachable") is False:
            state += " / 分支上也没有"
        elif reference.get("resolves_on"):
            state += " / 分支未取到"
        lines.append(
            f"| `{reference['referenced_by']}` | `{reference['path']}` | {reference['role']} | {state} | `{branch}` |"
        )
    lines.append("")

    families = manifest.get("related_problem_families") or []
    if families:
        lines.append("## 相邻但未纳入的题目族")
        lines.append("")
        lines.append("这些族有明显的 attention 成分，但不作为本 set 的条目，理由各自记录。")
        lines.append("")
        for family in families:
            lines.append(f"### `{family['family_id']}` — {family['problems']} 题")
            lines.append("")
            lines.append(f"- 位置：`{family['path']}`")
            lines.append(f"- 是什么：{family['what_it_is']}")
            lines.append(f"- 为什么不作为条目：{family['why_not_entries_here']}")
            lines.append(f"- 为什么仍然记录：{family['why_recorded']}")
            lines.append("")

    return "\n".join(lines) + "\n"


def _format_number(value: Optional[float]) -> str:
    if value is None:
        return "-"
    if value >= 1000:
        return f"{value:.1f}"
    return f"{value:.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="verify only; do not rewrite SET.md")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))

    facts: List[dict] = []
    problems: List[str] = []
    for entry in manifest["entries"]:
        entry_facts, entry_problems = verify_entry(entry)
        facts.append(entry_facts)
        problems.extend(entry_problems)

    dangling = verify_dangling(manifest["dangling_references"])
    for reference in dangling:
        if reference.get("resolves_on"):
            if reference["exists_locally"]:
                problems.append(
                    f"{reference['path']} is now in the working tree; "
                    f"the source archive is not supposed to be vendored here"
                )
            if reference.get("branch_reachable") is False:
                problems.append(
                    f"{reference['path']} is gone from {reference['resolves_on']}; "
                    f"{reference['referenced_by']} points at nothing"
                )
        elif reference["exists_locally"]:
            problems.append(
                f"{reference['path']} now exists locally; "
                f"{reference['referenced_by']} should be updated"
            )

    candidate_record_path = REPO_TOP / manifest["kernel_candidates_record"]
    if not candidate_record_path.is_file():
        problems.append(f"kernel candidate record missing at {manifest['kernel_candidates_record']}")
        candidates, record = [], {}
    else:
        record = load_candidate_record(manifest)
        candidates, candidate_problems = verify_candidates(manifest, record)
        problems.extend(candidate_problems)

    level1 = summarize_level1_archive()
    index = render_index(manifest, facts, dangling, level1, candidates, record)

    if not args.check:
        INDEX.write_text(index, encoding="utf-8")

    if problems:
        print("attention set drift detected:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    on_branch = sum(1 for reference in dangling if reference.get("resolves_on"))
    absent = len(dangling) - on_branch
    print(
        f"attention set OK: {len(manifest['entries'])} entries, "
        f"{sum(len(fact.get('measurements', [])) for fact in facts)} measurements verified, "
        f"{len(candidates)} kernel candidates, "
        f"{on_branch} source paths on the archive branch, {absent} absent everywhere"
    )
    if not args.check:
        print(f"wrote {INDEX.relative_to(REPO_TOP)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
