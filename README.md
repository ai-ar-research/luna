# Luna: Bound Propagator for Neural Network Analysis

## Introduction

Luna is a bound propagation engine for neural network verification written in C++20 with Python bindings. It implements state-of-the-art linear relaxation based perturbation analysis (LiRPA) algorithms, including
[CROWN](https://arxiv.org/pdf/1811.00866.pdf) and
[Alpha-CROWN](https://arxiv.org/pdf/2011.13824.pdf), to compute guaranteed
output bounds for neural networks under input perturbations.

Luna accepts models in [ONNX](https://onnx.ai/) format and property
specifications in [VNN-LIB](https://www.vnnlib.org/) format, making it
compatible with standard neural network verification benchmarks such as
[VNN-COMP](https://sites.google.com/view/vnn2025).

Our library supports the following bound propagation algorithms:

* Interval Bound Propagation ([IBP](https://arxiv.org/pdf/1810.12715.pdf))
* Backward mode LiRPA bound propagation ([CROWN](https://arxiv.org/pdf/1811.00866.pdf)/[DeepPoly](https://files.sri.inf.ethz.ch/website/papers/DeepPoly.pdf))
* Backward mode LiRPA with optimized bounds ([Alpha-CROWN](https://arxiv.org/pdf/2011.13824.pdf))

Luna supports a wide range of neural network layer types, including fully
connected, convolutional, transposed convolutional, batch normalization, ReLU,
sigmoid, reshape, flatten, slice, concatenation, and residual (add/sub)
connections.

The system description of Luna can be found [here](https://arxiv.org/abs/2603.23878).

## Technical Background

Neural network verification asks the question: given a neural network and a
bounded region of inputs, what are the guaranteed bounds on the network's
outputs?

**IBP** (Interval Bound Propagation) computes fast but loose bounds by
propagating intervals forward through each layer.

**CROWN** computes tighter bounds by propagating linear relaxations *backward*
through the computational graph, accumulating symbolic bound expressions that
are concretized at the input layer:

```
output >= A * input + b    (lower bound)
output <= A * input + b    (upper bound)
```

**Alpha-CROWN** further tightens CROWN bounds by introducing learnable slope
parameters (alpha) at nonlinear nodes (e.g., ReLU). These parameters are
optimized via gradient descent to minimize the bound gap:

```
for each iteration:
    run CROWN with current alphas
    compute loss = sum of bound widths
    update alphas via Adam optimizer
```

## Architecture

```
luna/
├── src/
│   ├── engine/
│   │   ├── TorchModel            # Main model class and computational graph
│   │   ├── CROWNAnalysis         # CROWN backward bound propagation
│   │   ├── AlphaCROWNAnalysis    # Alpha-CROWN optimization loop
│   │   └── nodes/                # Per-layer bound implementations
│   │       ├── BoundedLinearNode
│   │       ├── BoundedConvNode
│   │       ├── BoundedReLUNode
│   │       ├── BoundedSigmoidNode
│   │       └── ...               # 15+ node types
│   ├── input_parsers/
│   │   ├── OnnxToTorch           # ONNX model parser
│   │   └── VnnLibInputParser     # VNN-LIB property parser
│   └── configuration/
│       └── LunaConfiguration     # Static configuration settings
├── lunapy/                       # Python bindings (pybind11)
│   ├── lunapy.py                 # High-level Python API
│   └── examples/                 # Tutorial examples
├── tests/
│   ├── unit/                     # Google Test unit tests
│   ├── integration/              # End-to-end verification tests
│   └── property/                 # Invariant-based tests
└── resources/
    ├── onnx/                     # ONNX model files
    └── properties/               # VNN-LIB specifications
```

## Installation

### Requirements

- CMake 3.24+
- C++20 compiler
- PyTorch 2.5.1+ (Python package or standalone libtorch)

The following dependencies are fetched automatically via CMake FetchContent
when `--auto-download` is passed to `configure.sh`:

- Protobuf 3.19.2
- ONNX 1.15.0
- pybind11 2.11.1
- Google Test 1.15.2
- Google Benchmark 1.8.3

### Building

```bash
git clone <repository-url>
cd luna
./configure.sh --auto-download
make
```

### Running Tests

```bash
make test
```

### Configure Options

`configure.sh` accepts the following flags:

| Flag | Description |
|------|-------------|
| `--cpu` | Force CPU-only build (default) |
| `--gpu` | Enable CUDA libtorch |
| `--auto-download` | Fetch missing dependencies via FetchContent |
| `--debug` | Build in Debug mode |
| `--build-dir DIR` | Set build directory (default: `build`) |
| `--no-tests` | Disable test build |
| `--no-python` | Disable Python bindings |

GPU mode is auto-detected if `nvidia-smi` and a CUDA-enabled PyTorch are found.

### Make Targets

| Target | Description |
|--------|-------------|
| `make` | Build the project |
| `make test` | Run all tests |
| `make clean` | Clean build artifacts |
| `make install` | Install binaries |

### Building with GPU Support (HPC Cluster)

**1. Activate your Python environment with CUDA-enabled PyTorch:**

```bash
conda activate ab_env
```

**2. Request an interactive GPU node and load modules:**

```bash
srun -p gpu-a100-q --gres=gpu:1 --time=01:00:00 --pty bash
module load cmake-gcc11/3.21.3
module load cuda11.8/toolkit/11.8.0
```

**3. Configure and build:**

```bash
./configure.sh --gpu --auto-download
make
```

The build system auto-detects libtorch from your Python environment. If no
Python PyTorch is found and `--auto-download` is set, a pre-built libtorch
(CPU or CUDA) is fetched automatically.

> **Note:** The exact module names and partition name may differ depending on
> your cluster configuration.

## Quick Start

### Command Line

Run verification on an ONNX model with a VNN-LIB property specification:

```bash
luna --input model.onnx --vnnlib property.vnnlib
```

For example, verifying an ACAS Xu collision avoidance network:

```bash
luna --input resources/regular_benchmarks/benchmarks/acasxu_2023/onnx/ACASXU_run2a_1_1_batch_2000.onnx --property resources/regular_benchmarks/benchmarks/acasxu_2023/vnnlib/prop_3.vnnlib --method alpha-crown --optimize-lower --optimize-upper --lr 0.5 --iterations 20

```
