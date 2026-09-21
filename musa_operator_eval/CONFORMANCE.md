# Conformance with the guide

`摩尔线程musa算子构建指南.md` is the specification this set is built to. This file
records, requirement by requirement, where the repository satisfies it and where it
does not, so that a shortfall is a decision a reader can see rather than something
to be discovered later.

Every claim here is checked by something in `tests/` or by a command whose output is
quoted. Where a requirement is met by a declaration rather than by an enforcement,
that is said in the row.

## §1 The two tiers

| requirement | where |
|---|---|
| tier on every package | `task.json:tier`, `tests/test_task_packages.py` |
| the two tiers independent in task, API, baseline, metric, audit, ranking | `library_policy` versus `kernel_scope`; `scoring.ranking` says the tracks are never merged; `sets/attention/attention_set.json` records per-entry eligibility per tier |
| the three judgement questions | applied in the ledger's per-entry `tier.*.reason`, each citing a measurement |

## §2 What the set contains

| part | where | given to the agent |
|---|---|---|
| source snapshot, commit, licence | `sources/`, `task.json:source` | the provenance record only; the snapshot is fetched from the archive branch |
| environment snapshot | `environments/<snapshot_id>.public.json` (redacted), full copy under `private/environments/` | the redacted one |
| task contract, semantics, prompt, reference, public cases, starter | `tasks/<id>/` | yes |
| hidden cases, goldens, baselines | `private/<id>/`, `private/generated/<task id>/` | no |
| evaluator | `evaluator/` | not handed over; the scoring criteria are (`task.json:scoring`, `PROMPT.md`, `agent_reference/`) |

Checked by `evaluator/audit_agent_package.py` (all 11 packages) and
`tests/test_private_assets.py`.

**Shortfall, stated.** The evaluator's scripts are committed to this repository, so
anyone with the checkout can read them; a committed file cannot be un-published by a
later export step. `evaluator/README.md` records the production answer: mount that
directory from a private repository or an access-controlled store, the same way
`private/` is meant to be mounted.

## §3 Directory structure and the starter

`sources/`, `tasks/`, `private/`, `evaluator/` all exist. Each package carries
`starter/model_new.py` (the file the agent copies) and `starter/build.py` (the one
build entry point §4.7 runs), and `task.json:starter` is the §4.6 inventory.

The guide's starter table is written for a C++ project with CMake; this set's
submission is a Python module. The roles are realised as: build entry point
(`starter/build.py`, one file in every package, compared by a test), frozen header
and main program (the framework's `problem.py` / `ModelNew` contract), host launch
and device kernel (marked regions), dispatch file (a marked region — see §4.1 below),
tensor IO (the frozen `.bin` + JSON manifest ABI).

## §4.1 Task contract

Every §4.1 field is present, including the B tier's four extras. The dispatch file is
declared as `starter.dispatch`, because a submission here is one module and the
dispatch is a region of it; `library_policy` keeps the whitelist and the trace
requirements.

## §4.2 Semantics contract

Every package records the tensor signature, the dynamic-axis ranges, the ABI, the
determinism rule and §4.2's three pinned items — including the families where a mask
cannot arise, which say `not_applicable` rather than leaving the question open. The
two tiers of one entry share one semantic contract byte for byte, which
`tests/test_task_packages.py` checks; the graded dtype differs by tier and lives in
`tensor_contract`.

## §4.3 Case lists

Public: 3 smoke cases per package with golden paths recorded. Hidden: 4–10 cases per
package, private, covering correctness, boundary, non-aligned, extreme, performance
and a generalisation probe, with seeds disjoint from the public list and no repeated
shape. Cases carry parameters and seeds, not tensors; the generator is
`tools/generate_cases.py`.

**Shortfall, stated and argued.** §4.3 asks a B-tier hidden set to reach at least two
library-gap reasons. Four of the five B packages reach one (`unsupported_shape`); the
pilot reaches two (`unsupported_shape` and `semantic_mismatch`). Inventing a second
gap would mean grading a configuration the library actually serves.

The shortfall is not an omission, and it is now enforced rather than merely written
down: `private/<task id>/private_gap_evidence.json` carries an
`exactly_one_reason_category` block whose `why_the_others_are_out_of_reach` accounts
for every category the vocabulary has and is not reached, and
`tests/test_private_assets.py::GapCoverageTests` fails if a single-category task has
no such block or argues the wrong set of categories. The argument is per task because
the lever is per family. For the four non-pilot tasks it is the same one: the fused
path's only boundary is `head_dim <= 128` (the library says so itself: *"FlashAttention
in MUSA backend now requires head_dim to be less equal to 128, but got: 192"*), and an
isolating device probe recorded in each task's `probe_recheck` block shows the boundary
is a *route* rather than an error -- at head_dim 192 and 256 the call is answered on the
math path instead of raising -- and that float16, bfloat16 and float32 are all accepted
at head_dim 64 and 128, so the hypothesis that could have produced a second category
(the bfloat16 gap cases being dtype gaps) is refuted by measurement. What else holds:
the reference passes a frozen `1/sqrt(head_dim)` scale and a plain top-left causal mask
with no window (`semantic_mismatch` unreachable), hands the fused path the transposed
non-contiguous tensors it accepts (`layout_mismatch` unreachable), and asks for one
attention op whose neighbours in the block are elementwise work
(`multi_operator_required` unreachable — and it is a property of the task, not of a
case, so it cannot separate a gap case from an ordinary one).

**Shortfall, stated.** §4.3 asks the hidden shape range not to overlap the public one.
That holds for ten packages; `swin_transformer_v2_a_v0` cannot honour it, because the
reference partitions the image into windows after four downsampling stages and only a
few image sizes survive them. Its hidden cases move the batch instead, which the
manifest's `case_generation.note` says.

## Across the grades: one bar per task

The graded tolerance is resolved once, from the contract: `tolerances.max_abs_error`
is a loosen-only override of the framework's per-dtype floor
(`get_tolerance_for_precision`), applied to `atol` and `rtol` alike, and both drivers
ask the same question of the same declaration. `run_binary.py` replicates the rule
rather than importing torch, and a device test compares the replica against
`get_tolerance_for_precision` and `resolve_tolerance` so a change there fails a test.

This replaced a real inconsistency: the hidden manifests carried a per-case
`atol`/`rtol` (0.01 or 0.02) that only the compiled driver read, while the Python path
graded at 0.05 -- two bars for one task, where a submission at error 0.03 passed one
path and failed the other. The per-case field is gone from the schema and from all
twelve manifests, `additionalProperties: false` makes a leftover key invalid, and every
report row and every report now records the `tolerance_used`.

## §4.4 Baselines

Every package has `private/<task id>/baseline.hidden.json` measured on the target
environment, with the machine fingerprint, the measurement protocol, per-case
latency, `p10`/`p90`, throughput, peak memory and status, and a `scoring` block
naming the denominator: `upstream_musa` for the A tier, `expert_dispatch` for the B
tier.

**One protocol, and it is the framework's.** A baseline's `latency_ms` is a denominator
and a `run_task.py` report's `runtime` is a numerator, so the two have to be the same
kind of number. They were not: the tool carried its own protocol -- warmup 10, five
rounds of a hundred calls, the median of the round means, no cache flush -- while the
reports come from `kernelbench.timing`, whose loop warms up, empties the allocator, then
runs one pass of a hundred trials with an L2 thrash before each (so its numbers are
cold-cache numbers), discards the first trial and reports the **mean**. Every ratio was
consequently a ratio between two protocols, and the fact that they agreed to within a
tenth was luck rather than design.

The tool now calls that loop instead of imitating it:
`kernelbench.timing.time_execution_with_cuda_event` measures, `get_timing_stats`
summarises, and the record's `measurement_protocol` names both. `latency_ms` is the mean
of the trials, `latency_ms_samples` is the trials themselves (renamed from
`latency_ms_rounds`, because the framework's loop has one pass and calling a trial a
round described a structure it does not have), and `latency_ms_stats` keeps the spread
the mean came from. `tests/test_private_assets.py` fails if a record does not name that
loop or if its number is not the mean of its own samples.

The A tier's roster is unchanged: upstream implementation plus `reference_model`, with
the library paths recorded as `unavailable` for the reason that tier has.

The numbers moved when the protocol did, most of all on the small shapes where a flushed
cache costs the most, and no decision moved with them:

| package | before | after |
|---|---|---|
| `scaled_dot_product_attention_b_v0` (level 1, B) | 3.834 | 3.213 |
| `vision_attention_b_v0` | 1.751 | 1.712 |
| `mingpt_causal_attention_b_v0` | 3.691 | 3.447 |
| `mingpt_block_b_v0` | 2.190 | 2.124 |
| `sdpa_forward_b_v0` (the pilot) | 7.811 | 6.152 |

**Shortfall, stated.** Not every roster entry could be built honestly here. The A
tier records `mudnn_fused` and `torch_musa_sdpa` as `unavailable` (the tier is graded
at float32, where the fused path does not answer), and the four entry-scoped B
packages record `upstream_musa`, `mudnn_fused` and `torch_musa_eager` as
`unavailable`. The measurement tool records an implementation it cannot honestly
build as unavailable with a reason rather than substituting a lookalike; a baseline
under a real implementation's name would be worse than a gap.

Also worth recording: the A tier's upstream implementation refuses most hidden shapes
with `RuntimeError: Tile divisibility` and, for `vision_attention_a_v0`, agrees with
the reference on only four of eight cases. Those rows are in the baseline records.
The hand-written answers are specialised to the shapes they were tuned on, which is
what a hidden set outside the public range is for.

The B tier's gate (naive-to-expert ratio, geometric mean, floor 1.3) is re-measured
and re-recorded by `measure_baseline.py` → `candidate_gate.py` →
`record_admission.py`. All five admitted packages reproduce their decisions; the
per-case evidence lives in `private/<task id>/admission.json` and the contract keeps
the outcome only, because §2 hands the contract to the agent.

## §4.5 Prompts

Seven sections per package, with the tier direction asserted by test: the A tier's
prohibitions name the library entry points, the B tier's requirements say the library
path comes first. No prompt names the upstream project, a library function or a
version string; the budget matches the contract.

## §4.6 Starter inventory and ABI

`task.json:starter` declares the editable file and its marked regions, the frozen
list, the build and self-test commands, the ABI and — for the B tier — the link
whitelist, which a test requires to equal `library_policy`. The static audit reads
`frozen` from there, so the boundary it draws is the declared one.

**Met on the compiled path.** §4.6 asks the runner not to import libtorch, so ATen
cannot be reached from the graded process. Every package ships `starter/cpp`: a
frozen `runner.cc` and `abi_io.h` that read `<input-dir>/tensors.json` and its
`.bin` files, call `kernel_entry`, and write `<output-dir>/tensors.json` and its
`.bin` files; a frozen `build.sh` that compiles with `mcc` against the MUSA runtime
and `--extra-library` for whatever the contract whitelists; and one editable
`kernel.mu`. The runner's dynamic section names `libmusart` and the C runtime and
no torch, which is what makes the sentence true of the graded process.

`evaluator/run_binary.py` drives it through §4.7's stages against the same goldens.
On the device, `private/scaled_dot_product_attention_a_v0/compiled/kernel.mu` — a
worked submission kept under `private/` because it is an answer — passes all nine
hidden cases with a worst absolute error of 2.2e-06 against the 0.05 bar,
over stages `environment, case_generation, static_audit, build, evaluation`.

**Deviation, recorded, Python path only.** The Python path remains a module whose
`forward` is called in the evaluator's own process, which is where ATen lives, so
the requirement is not met there and `starter.abi.deviation` says so in every
package. The compiled path is the compliant one; the Python path is kept for the
families whose dispatch is only reachable through the framework's bindings.

**Met on the compiled path for the level-1 family, and verified end to end.**
muDNN's host API is present on this machine and needs no torch: the toolkit ships
`mudnn.h`, `mudnn_base.h`, `mudnn_math.h` and `mudnn_nn.h` under
`/usr/local/musa-3.1.0/include/` (`$MUSA_HOME/include`), and `mudnn_nn.h` declares
`musa::dnn::ScaledDotProductAttention` with `SetHeadsNum`, `SetEmbedDim`, `SetCausal`,
`SetMaskMode`, `SetKeyFormat`, `RunFlash` and `RunMath`. `libmudnn.so` exports 10001
`musa::dnn::*` symbols, and a program that links only `libmudnn` and `libmusart` --
no torch in its dynamic section -- gets `SUCCESS` back from `RunFlash`.

`private/scaled_dot_product_attention_b_v0/compiled/kernel.mu` is that adapter for
this family. It asks the library first and reads the answer rather than predicting
it: on this build the fused kernel answers head dimension up to 160 and refuses above
it (*"Flash Attention 2 Not Support HeadDim > 160 Now"*), so the adapter takes the
fused path or the library's non-fused path and records which one produced the number.
Through `evaluator/run_binary.py` over the ten hidden cases it exits 0 with 10 of 10
passed, every recorded path equal to its expected one, and the artifact's dynamic
section naming `libmudnn` and no torch.

**The compiled path's input set, and what decides it.** It grades a case out of the
tensors in its input directory -- that is the whole interface -- and eight of the
eleven packages are references that build weights in `__init__`, which a torch-free
submission cannot reconstruct: they are a function of the case's seed and the
construction order, and §4.3 deliberately keeps a case to "parameters and seeds, not
tensors". So the evaluator materializes that state with the framework's own
construction -- seed, `get_init_inputs()`, seed, `Model(*init)`, cast to the case's
dtype, the order `kernelbench.eval` uses -- stages it beside the case's own tensors
under `state_*` names, and records a digest of the whole state on every case row.

The generated tree is never written to: the runner reads a staged copy, so the case's
own inputs stay what they were and the Python path, which builds its own model, reads
nothing new. No golden was regenerated for this, because the state comes from the same
seed stream the golden was produced from. `task.json:starter.cpp.model_state` declares
the arrangement per package (`tensor_prefix`, `materialized_by`, where the seeding
comes from) or is null for a reference that carries nothing.

The scope is therefore a property of what each task hands over rather than of the
tiers: `kb_l1_97_a` and `kb_l1_97_b` are graded from their three tensors alone, and the
other nine from their inputs plus staged state. What is still missing for the
dispatch-shaped families is not the mechanism but the submissions -- a compiled
adapter that routes projection, attention and the surrounding block through muDNN
operators -- and those are per-family work, verified on the device like the level-1
one. `tests/test_private_assets.py::CompiledAnswerScopeTests` fails if an answer exists
where neither the case's inputs nor a declaration covers it, and
`tests/test_materialize_model_state.py` checks on the device that each declaration
matches what its reference actually holds.

The anti-cheating half holds independently of that: the artifact is built, its
dynamic section is read, and a submission linking outside the whitelist is refused
before a single case runs (a device test compiles a program that really links zlib and
asserts the run stops with `EXIT_LINK_WHITELIST_VIOLATION`).

**Met, and checked, on the compiled path.** §4.6 confines library calls to the
dispatch file and the Host launch file. A Python submission is a single module, so
that is a region constraint rather than a file constraint; a compiled one marks its
regions in the source, so `run_binary.audit_compiled_source` reads the markers and
rejects a library call outside the permitted regions — and, on the A tier, any
reference to the upstream stacks at all. The check is what rejected an early
revision of the worked kernel for declaring one region where the inventory
declared two.

## §4.7 Evaluator

Nine stages, short-circuiting, named exit codes: environment, case generation, build,
static audit, link whitelist (B only), dispatch trace (B only, per case), correctness
with the stability re-run, performance, summary. `run_task.py` reports the stages it
ran; a stage that fails is not followed by the next one, which the device runs show
as `skipped` rows.

The same chain is driven over the compiled path by `evaluator/run_binary.py`, which
runs the package's `starter/cpp/build.sh`, checks what the artifact links, runs the
runner per case and compares its tensors against the same goldens. On the device it
reports `[environment, case_generation, static_audit, build, link_whitelist_check,
evaluation]` for a B-tier task, and the link stage reads three artifacts including
the linked `runner` itself (`libmusart.so.1.0`, `libmusa.so.1.0`, `libmudnn.so.2`
plus plumbing, 69 undefined and 10 defined symbols) — which is the first time this
stage has had a real B-tier artifact to check rather than "no compiled artifact was
produced".

The link check parses the build products' dynamic sections with no dependency beyond
the standard library, and was checked against `readelf -d` and `nm -D` on six real
MUSA extension objects and on system binaries: identical dependency and symbol sets.

**Timing, and what `runtime` means on this path.** A compiled case runs in a process
that starts cold: measured on the target device, the first call in a fresh process pays
237 ms for the device context and roughly as much again for mapping a 2.2 GB muDNN,
the case's files cost 58-470 ms to read and write depending on size, and the attention
call itself takes 0.5 ms the first time and under 0.1 ms thereafter. A wall clock around
that process is therefore 99.9% startup and I/O, and comparing it with the Python
path's `runtime` -- which is measured in a warm process with the tensors already on the
device -- compares two different quantities.

So the frozen runner takes `--warmup W --repeat N` (defaults 3 and 100, the framework's
own call counts): it reads the inputs once, calls `kernel_entry` W times to settle the
process and the device, then takes `repeat + 1` timed calls and reports each one. The
clock is in the frozen half, so a submission cannot report its own grade, and the
statistic is in the driver, where a test can hold it.

**The measurement is the framework's protocol, not one this path invented.** The same
commit that moved the baselines onto `kernelbench.timing` moved the runner onto the same
measurement, so that a compiled `runtime` and a baseline's `latency_ms` divide each other
as like by like: a MUSA **event pair** around each call rather than a host clock, the
**L2 cache flushed before every timed call** (a 256 MB fill, which is what upstream's
`clear_l2_cache` is) so the numbers are cold-cache numbers, the **first timed call
discarded** as `discard_first=1` discards it, and the **mean** of what is left, in the
three-significant-digit format `get_timing_stats` renders. The flush is queued outside
the measured interval, and the fields that describe the protocol travel in the row --
`timer`, `cache`, `l2_thrash_bytes`, `samples` -- so a row cannot claim a measurement it
did not make: a runner that could not allocate the flush buffer records `cache: "warm"`,
and a runner that reports no timings at all still grades, with `runtime` falling back to
the process wall clock and `timing` null, which is what an older starter does.

Two boundaries are stated rather than left to be discovered. The events are recorded on
the device's default stream, so a submission that launches on a stream of its own and
does not synchronize is timed as if it had done nothing; every worked answer and every
skeleton uses the default stream, which is the pattern the ABI documents. And a compiled
case is still only ever ranked against another compiled case, because the forms differ by
their overheads -- the ABI's host-to-device copies are inside this interval by design --
rather than by their clocks.

**What aligning the two clocks cost, measured.** Both compiled rows were re-measured
under the framework's protocol (event pair, per-call L2 flush, first trial discarded,
mean) and both still pass -- the worked A kernel 9 of 9, the B adapter 10 of 10 -- but
they moved by different amounts, and the difference is the interesting part. The worked
kernel's smallest case went from 3.8 ms to 10.7 ms, a factor of 2.8 with a *tight*
distribution (100 samples spanning 10.68-10.84 ms), because that kernel is compute- and
cache-bound and a flushed cache is exactly what it feels. The B adapter barely moved:
0.501 ms to 0.496 ms on the single-token case, 24.356 ms to 23.8 ms on the widest one,
because those rows are the ABI's host-to-device transport rather than cache behaviour
(§4.7's arithmetic: 98% of that case is the copy). A protocol change that had only moved
the small case would have been a change in the timer; the fact that it moved the
cache-sensitive row and left the transport-bound row alone is evidence it did what it
says.

**Defect found and fixed by that run.** The stage selected build products by suffix
(`.so`, `.o`), and a linker's final product is an executable with no suffix at all,
so it was reading exactly the artifacts that have no dynamic section and reporting a
pass. It now identifies products by the ELF magic and keeps the suffix list as a
second signal, so a mangled object is still reported rather than skipped. Three tests
cover it, including that a suffixless executable is checked and its libraries read.

## §4.8 Schema validation

Six schemas, versioned, validated by value with the tier rules. `validate_schemas.py
--all` reports 68/68 valid on the maintainer tree and 34/34 on a checkout with
`private/` emptied, and all thirteen test files pass in both: the ones that read
maintainer assets -- the hidden lists, the baselines and the gap evidence -- skip
there (`test_private_assets` runs 19 tests, 8 skipped) rather than failing, which is
what a checkout without those files should report.

## §5 Build steps

Environment fixed and compared before grading; sources recorded; semantics extracted
per family with an independent CPU oracle for the family that has one; contracts and
prompts written per tier; cases generated per tier; one interface per tier; baselines
measured and gated; the evaluator assembled and the tracks kept separate.

**Met.** §5 step 8's "separately split, separately ranked" is enforced in two places:
the scoring blocks and case lists never mix tiers, and `evaluator/rank.py` produces
the two tables -- never a combined figure, because a single number across a device
kernel and a library dispatch is a number about nothing. Each ratio is the tier's
own denominator, read from the baseline's `scoring.speedup_denominator`
(`upstream_musa` for A, `expert_dispatch` for B), computed per case over the cases
where the reference has a passing measurement on the same case, and summarised as a
geometric mean. Both sides of a ratio are milliseconds, and since §4.4's change both
come from the framework's own timing loop, so a ratio is between two implementations
rather than between two protocols. The unit is the one the baseline is in: the
framework's dataclass comment says microseconds and the recorded values say otherwise,
so `HARDWARE_RUNBOOK.md` records the trap.

The device readings, each against the baseline measured the framework's way:

| report | ratio | cases compared |
|---|---|---|
| A tier: the worked compiled kernel (fp32, steady state) | `x0.05306` | 2 of 9 |
| B tier: the compiled muDNN adapter | `x0.08565` | 10 of 10 |
| B tier: the five Python experts | `x0.99586` to `x1.00842` | 48 of 48 |

Three readings from that session are worth keeping. A B-tier Python report now lands
within 0.4% of its own baseline, because the report and the baseline are the same
expert measured by the same loop; the same comparison read `x0.9112` and `x0.892` while
the two sides were measured differently, and that nearness to 1 was luck rather than a
property of the measurement. The compiled adapter reads `x0.08007`, twelve and a half
times behind the expert, which is the host-side staging the ABI asks a submission to do
rather than the library call. And the A tier's worked kernel reads `x0.05187`, about
twenty times slower than the upstream answer -- where the same kernel measured through a
process wall clock had read `x0.002615`, four hundred times slower, because that number
was process startup, device initialisation and disk. The last pair is why a compiled case
is only ever ranked against another compiled case: the forms differ by their overheads,
and the overheads are now visible instead of being the measurement.
