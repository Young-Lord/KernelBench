# Private evaluation assets

Everything below this directory is evaluator-only and is ignored by Git.
Do not place hidden cases, golden tensors, expert dispatch source, or baseline
artifacts in an agent-visible checkout.

For production, store this tree in a separate private repository or an
access-controlled artifact store and mount it only into the evaluator.
