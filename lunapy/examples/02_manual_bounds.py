#!/usr/bin/env python3
"""
Example 2: Manual Bound Setting

This example shows how to:
- Load an ONNX model without VNN-LIB
- Set input bounds manually
- Compute bounds with different methods
"""

from lunapy import TorchModel
import numpy as np
from pathlib import Path

def main():
    print("="*60)
    print("Example 2: Manual Bound Setting")
    print("="*60)

    repo_root = Path(__file__).parent.parent.parent
    onnx_path = str(repo_root / "resources" / "onnx" / "spec_test.onnx")

    print(f"\n1. Loading ONNX model (without VNN-LIB)...")
    model = TorchModel(onnx_path)

    print(f"   Model loaded: {model.getInputSize()} inputs, {model.getOutputSize()} outputs")
    print(f"   Has input bounds: {model.hasInputBounds()}")

    # Create input bounds manually
    input_size = model.getInputSize()

    print(f"\n2. Setting input bounds manually...")

    # Example 1: Uniform bounds [-1, 1]
    print(f"   Creating uniform bounds: [-1.0, 1.0] for all {input_size} inputs")
    lower_bounds = np.full(input_size, -1.0, dtype=np.float32)
    upper_bounds = np.full(input_size, 1.0, dtype=np.float32)

    model.setInputBounds(lower_bounds, upper_bounds)
    print(f"   Bounds set successfully")
    print(f"   Has input bounds: {model.hasInputBounds()}")

    # Verify bounds
    retrieved_bounds = model.getInputBounds()
    print(f"   Verified: lower[0] = {retrieved_bounds.lower()[0]}, upper[0] = {retrieved_bounds.upper()[0]}")

    print(f"\n3. Computing bounds with CROWN...")
    result_crown = model.compute_bounds(method='CROWN')

    print(f"   CROWN results:")
    print(f"     Lower: {result_crown.lower()}")
    print(f"     Upper: {result_crown.upper()}")

    # Example 2: Different bounds
    print(f"\n4. Trying different input bounds...")
    print(f"   Setting tighter bounds: [-0.5, 0.5]")

    lower_bounds = np.full(input_size, -0.5, dtype=np.float32)
    upper_bounds = np.full(input_size, 0.5, dtype=np.float32)
    model.setInputBounds(lower_bounds, upper_bounds)

    result_tight = model.compute_bounds(method='CROWN')

    print(f"   CROWN results with tighter bounds:")
    print(f"     Lower: {result_tight.lower()}")
    print(f"     Upper: {result_tight.upper()}")

    # Compare widths
    width_original = np.sum(result_crown.upper() - result_crown.lower())
    width_tight = np.sum(result_tight.upper() - result_tight.lower())

    print(f"\n5. Comparison:")
    print(f"   Total width with [-1, 1] bounds: {width_original:.6f}")
    print(f"   Total width with [-0.5, 0.5] bounds: {width_tight:.6f}")
    print(f"   Reduction: {100 * (1 - width_tight/width_original):.2f}%")

    # Example 3: Random asymmetric bounds
    print(f"\n6. Asymmetric bounds example...")

    np.random.seed(42)
    lower_bounds = np.random.uniform(-2.0, -0.5, input_size).astype(np.float32)
    upper_bounds = np.random.uniform(0.5, 2.0, input_size).astype(np.float32)

    print(f"   Lower bounds range: [{lower_bounds.min():.3f}, {lower_bounds.max():.3f}]")
    print(f"   Upper bounds range: [{upper_bounds.min():.3f}, {upper_bounds.max():.3f}]")

    model.setInputBounds(lower_bounds, upper_bounds)
    result_asym = model.compute_bounds(method='CROWN')

    print(f"   CROWN results:")
    print(f"     Lower: {result_asym.lower()}")
    print(f"     Upper: {result_asym.upper()}")

    print("\n" + "="*60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
