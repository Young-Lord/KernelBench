#!/bin/bash
# The compiled starter's one build path (§4.7's build stage runs this and nothing
# else). It links the MUSA runtime and, for the B tier, the contract's whitelisted
# libraries -- and libtorch is not among them, which is what keeps ATen out of the
# process the submission runs in.
#
#   ./build.sh --submission <dir-with-kernel.mu> --build-dir <dir>
set -euo pipefail

MUSA_HOME="${MUSA_HOME:-/usr/local/musa}"
ARCH="${MUSA_ARCH:-mp_22}"
SUBMISSION=""
BUILD_DIR="build"

while [ $# -gt 0 ]; do
  case "$1" in
    --submission) SUBMISSION="$2"; shift 2 ;;
    --build-dir)  BUILD_DIR="$2";  shift 2 ;;
    --arch)       ARCH="$2";       shift 2 ;;
    --extra-library) EXTRA_LIBRARIES="${EXTRA_LIBRARIES:-} -l$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [ -z "$SUBMISSION" ]; then
  SUBMISSION="$(cd "$(dirname "$0")" && pwd)"
fi
mkdir -p "$BUILD_DIR"
BUILD_DIR="$(cd "$BUILD_DIR" && pwd)"
STARTER="$(cd "$(dirname "$0")" && pwd)"

echo "[build] mcc ${MUSA_HOME}/bin/mcc, arch ${ARCH}"
"${MUSA_HOME}/bin/mcc" -x musa -O3 "--offload-arch=${ARCH}" -c "${SUBMISSION}/kernel.mu" \
  -o "${BUILD_DIR}/kernel.o" -I"${STARTER}" -I"${MUSA_HOME}/include"
"${MUSA_HOME}/bin/mcc" -O3 -std=c++14 -c "${STARTER}/runner.cc" \
  -o "${BUILD_DIR}/runner.o" -I"${STARTER}" -I"${MUSA_HOME}/include"
"${MUSA_HOME}/bin/mcc" -O3 "${BUILD_DIR}/runner.o" "${BUILD_DIR}/kernel.o" \
  -o "${BUILD_DIR}/runner" -L"${MUSA_HOME}/lib" -lmusart ${EXTRA_LIBRARIES:-} \
  -Wl,-rpath,"${MUSA_HOME}/lib"
echo "[build] ${BUILD_DIR}/runner"
