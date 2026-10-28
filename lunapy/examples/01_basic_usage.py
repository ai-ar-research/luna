#!/usr/bin/env python3
"""
Example 1: Basic Usage

This example demonstrates the most basic usage of lunapy:
- Loading an ONNX model with VNN-LIB bounds
- Computing bounds using CROWN
- Accessing the results
"""

from lunapy import TorchModel
import numpy as np
from pathlib import Path

def main():
    print("="*60)
    print("Example 1: Basic Usage")
    print("="*60)

    # Paths to test resources (relative to repo root)
    repo_root = Path(__file__).parent.parent.parent
    onnx_path = str(repo_root / "resources" / "onnx" / "spec_test.onnx")
    vnnlib_path = str(repo_root / "resources" / "properties" / "spec_test.vnnlib")

    print(f"\n1. Loading model...")
    print(f"   ONNX: {onnx_path}")
    print(f"   VNN-LIB: {vnnlib_path}")

    # Create model (bounds loaded automatically from VNN-LIB)
    model = TorchModel(onnx_path, vnnlib_path)

    print(f"\n2. Model information:")
    print(f"   Input size: {model.getInputSize()}")
    print(f"   Output size: {model.getOutputSize()}")
    print(f"   Number of nodes: {model.getNumNodes()}")
    print(f"   Has input bounds: {model.hasInputBounds()}")

    # Check what bounds were loaded
    if model.hasInputBounds():
        bounds = model.getInputBounds()
        print(f"\n3. Input bounds from VNN-LIB:")
        print(f"   Lower: {bounds.lower()[:5]}... (showing first 5)")
        print(f"   Upper: {bounds.upper()[:5]}... (showing first 5)")

    print(f"\n4. Running CROWN analysis...")

    # Compute output bounds using CROWN
    result = model.compute_bounds(method='CROWN')

    print(f"\n5. Results:")
    print(f"   Output lower bounds: {result.lower()}")
    print(f"   Output upper bounds: {result.upper()}")
    print(f"   Bound width: {result.upper() - result.lower()}")

    print(f"\n6. Analysis complete")
    print("="*60)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"Error: Could not find test files: {e}")
        print("Make sure you're running from the lunapy/examples directory")
        print("and that resources directory exists in the repository root.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
