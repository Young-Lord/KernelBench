#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
session_name=cudallm-musa
host=${CUDALLM_HOST:-127.0.0.1}
port=${CUDALLM_PORT:-8000}
cache=${CUDALLM_CACHE:-static}
log_file="$repo_dir/logs/cudallm-musa-screen.log"

is_running() {
    screen -ls 2>/dev/null | grep -q "[.]${session_name}[[:space:]]"
}

start() {
    if is_running; then
        echo "cudaLLM is already running at http://${host}:${port}"
        return 0
    fi
    mkdir -p "$repo_dir/logs"
    screen -L -Logfile "$log_file" -dmS "$session_name" bash -lc \
        "cd '$repo_dir' && exec scripts/serve_musa.sh --host '$host' --port '$port' --cache '$cache'"
    for _ in 1 2 3 4 5 6 7 8 9 10 11 12; do
        if curl -fsS "http://${host}:${port}/health"; then
            printf '\ncudaLLM started in screen session %s\n' "$session_name"
            return 0
        fi
        sleep 5
    done
    echo "cudaLLM failed to become ready; inspect $log_file" >&2
    return 1
}

stop() {
    if ! is_running; then
        echo "cudaLLM is not running"
        return 0
    fi
    screen -S "$session_name" -X quit
    echo "cudaLLM stopped"
}

status() {
    if is_running; then
        curl -fsS "http://${host}:${port}/health"
        printf '\n'
    else
        echo "cudaLLM is not running"
        return 1
    fi
}

case ${1:-status} in
    start) start ;;
    stop) stop ;;
    restart) stop; start ;;
    status) status ;;
    logs) tail -f "$log_file" ;;
    *) echo "Usage: $0 {start|stop|restart|status|logs}" >&2; exit 2 ;;
esac
