# lunapy Examples

This directory contains example scripts demonstrating various features of lunapy.

## Running Examples

**First, install the package:**
```bash
cd /Users/hlecates/Desktop/autoluna/luna
pip install -e .
```

**Then run examples from any directory:**
```bash
# From the examples directory
cd lunapy/examples
python 01_basic_usage.py
python 02_manual_bounds.py
python 03_alpha_crown.py
python 04_specification_matrix.py
python 05_forward_pass.py

# Or from the repo root
python lunapy/examples/01_basic_usage.py
python lunapy/examples/02_manual_bounds.py
# ... etc
```

## Descriptions

### 01_basic_usage.py
**Basic Usage** - The simplest way to use lunapy
- Loading ONNX model with VNN-LIB bounds
- Computing bounds with CROWN
- Accessing results

### 02_manual_bounds.py
**Manual Bound Setting** - Setting input bounds programmatically
- Loading ONNX without VNN-LIB
- Setting bounds manually with NumPy
- Comparing results with different bounds

### 03_alpha_crown.py
**Alpha-CROWN Optimization** - Using optimized bounds
- Comparing CROWN vs Alpha-CROWN
- Configuring optimization parameters
- Performance analysis

### 04_specification_matrix.py
**Specification Matrices** - Computing bounds on output combinations
- Identity specifications
- Pairwise differences (classification)
- One-vs-all verification
- Custom weighted combinations

### 05_forward_pass.py
**Forward Pass and Verification** - Testing concrete inputs
- Performing forward passes
- Verifying outputs are within bounds
- Random sampling
- Bound tightness analysis

## Prerequisites

All examples assume you have:
1. Built and installed lunapy (`pip install -e .`)
2. Test ONNX models in `../../resources/onnx/`
3. Test VNN-LIB files in `../../resources/properties/`
