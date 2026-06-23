#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="build"
BUILD_TYPE="Release"
GPU_MODE="OFF"
AUTO_DOWNLOAD="OFF"
BUILD_TESTS="ON"
BUILD_PYTHON="ON"
EXTRA_CMAKE_ARGS=""

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  --cpu              Force CPU-only build (default)
  --gpu              Enable CUDA libtorch
  --auto-download    Download dependencies via FetchContent when missing
  --debug            Build in Debug mode
  --build-dir DIR    Set build directory (default: build)
  --no-tests         Disable test build
  --no-python        Disable Python bindings
  -h, --help         Show this help
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpu)          GPU_MODE="OFF"; shift ;;
        --gpu)          GPU_MODE="ON"; shift ;;
        --auto-download) AUTO_DOWNLOAD="ON"; shift ;;
        --debug)        BUILD_TYPE="Debug"; shift ;;
        --build-dir)    BUILD_DIR="$2"; shift 2 ;;
        --no-tests)     BUILD_TESTS="OFF"; shift ;;
        --no-python)    BUILD_PYTHON="OFF"; shift ;;
        -h|--help)      usage ;;
        *)              EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS $1"; shift ;;
    esac
done

# Auto-detect GPU if --gpu/--cpu not explicitly set
if [[ "$GPU_MODE" == "OFF" ]]; then
    if command -v nvidia-smi &>/dev/null; then
        if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
            echo "Auto-detected CUDA-enabled PyTorch, enabling GPU mode"
            GPU_MODE="ON"
        fi
    fi
fi

# Prefer Ninja if available
GENERATOR_ARGS=""
if command -v ninja &>/dev/null; then
    GENERATOR_ARGS="-G Ninja"
fi

echo "=== Luna Build Configuration ==="
echo "  Build directory: ${BUILD_DIR}"
echo "  Build type:      ${BUILD_TYPE}"
echo "  GPU mode:        ${GPU_MODE}"
echo "  Auto-download:   ${AUTO_DOWNLOAD}"
echo "  Tests:           ${BUILD_TESTS}"
echo "  Python bindings: ${BUILD_PYTHON}"
if [[ -n "$GENERATOR_ARGS" ]]; then
    echo "  Generator:       Ninja"
else
    echo "  Generator:       Make"
fi
echo "================================="

cmake -B "${BUILD_DIR}" \
    ${GENERATOR_ARGS} \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DLUNA_GPU="${GPU_MODE}" \
    -DLUNA_AUTO_DOWNLOAD="${AUTO_DOWNLOAD}" \
    -DBUILD_TESTS="${BUILD_TESTS}" \
    -DBUILD_PYTHON_BINDINGS="${BUILD_PYTHON}" \
    ${EXTRA_CMAKE_ARGS}

echo ""
echo "Configuration complete. Next steps:"
echo "  make            # build"
echo "  make test       # run tests"
echo "  make clean      # clean build"
