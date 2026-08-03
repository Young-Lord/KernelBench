#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export MUSA_HOME=${MUSA_HOME:-/usr/local/musa}
export PATH="$MUSA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$MUSA_HOME/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME=${HF_HOME:-/root/autodl-tmp/huggingface}

exec /root/miniconda3/envs/kernelbench-musa/bin/python \
    "$repo_dir/serve_musa.py" "$@"
