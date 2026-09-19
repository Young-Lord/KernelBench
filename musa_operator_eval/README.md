# MUSA Operator Evaluation

This directory turns `docs/musa_operator_evaluation_guide.md` into an executable,
versioned evaluation format. Version 0 is intentionally scoped to Attention
families. Other operator families must add their own semantic contract rather
than reusing Attention-specific fields.

## Layout

- `schemas/`: versioned JSON Schema documents, including the agent-reference format.
- `fixtures/`: public positive and negative schema fixtures.
- `tools/`: dependency-free validation and generation utilities.
- `tests/`: CPU-only unit tests for the control plane.
- `progress/`: one Markdown hand-off for every completed phase.
- `tasks/`: agent-visible task packages, added in later phases.
- `agent_reference/`: sanitized, agent-visible capability descriptions.
- `private/`: evaluator-only assets; ignored by Git except for its README.

## Validate the current control plane

```bash
python musa_operator_eval/tools/validate.py --fixtures
python -m unittest discover -s musa_operator_eval/tests -v
```

The validator uses only the Python standard library. The schemas remain valid
JSON Schema 2020-12 documents so they can also be consumed by a full JSON
Schema implementation in CI later.
