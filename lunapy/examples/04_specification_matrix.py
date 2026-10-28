#!/usr/bin/env python3
"""
Example 4: Specification Matrices

This example demonstrates:
- Using custom specification matrices
- Computing bounds on output combinations
- Verification use cases
"""

from lunapy import TorchModel
import numpy as np
from pathlib import Path

def main():
    print("="*60)
    print("Example 4: Specification Matrices")
    print("="*60)

    repo_root = Path(__file__).parent.parent.parent
    onnx_path = str(repo_root / "resources" / "onnx" / "spec_test.onnx")
    vnnlib_path = str(repo_root / "resources" / "properties" / "spec_test.vnnlib")

    print(f"\n1. Loading model...")
    model = TorchModel(onnx_path, vnnlib_path)

    output_size = model.getOutputSize()
    print(f"   Model has {output_size} outputs")

    # Example 1: Identity matrix (same as no specification)
    print(f"\n2. Identity specification (equivalent to no spec)...")

    C_identity = np.eye(output_size, dtype=np.float32)
    result_identity = model.compute_bounds(method='CROWN', C=C_identity)

    print(f"   Specification matrix shape: {C_identity.shape}")
    print(f"   Result lower: {result_identity.lower()}")
    print(f"   Result upper: {result_identity.upper()}")

    # Compare with no specification
    result_no_spec = model.compute_bounds(method='CROWN')
    print(f"\n   Verification: results match no-spec case")
    print(f"   Max difference: {np.max(np.abs(result_identity.lower() - result_no_spec.lower())):.8f}")

    # Example 2: Pairwise differences (for classification)
    if output_size >= 2:
        print(f"\n3. Pairwise output differences (classification robustness)...")
        print(f"   Computing bounds on: output[i] - output[i+1]")

        C_diff = np.zeros((output_size - 1, output_size), dtype=np.float32)
        for i in range(output_size - 1):
            C_diff[i, i] = 1.0
            C_diff[i, i + 1] = -1.0

        print(f"   Specification matrix shape: {C_diff.shape}")
        print(f"   Matrix:\n{C_diff}")

        result_diff = model.compute_bounds(method='CROWN', C=C_diff)

        print(f"\n   Results:")
        for i in range(output_size - 1):
            lower = result_diff.lower()[i]
            upper = result_diff.upper()[i]
            print(f"   output[{i}] - output[{i+1}]: [{lower:.6f}, {upper:.6f}]")

            if lower > 0:
                print(f"     ✓ Verified: output[{i}] > output[{i+1}]")
            elif upper < 0:
                print(f"     ✓ Verified: output[{i}] < output[{i+1}]")
            else:
                print(f"     ? Unknown: bounds cross zero")

    # Example 3: Target class vs all others (one-vs-all)
    if output_size >= 3:
        print(f"\n4. One-vs-all specification (target class 0)...")
        print(f"   Computing: output[0] - output[i] for i != 0")

        C_ova = np.zeros((output_size - 1, output_size), dtype=np.float32)
        for i in range(1, output_size):
            C_ova[i - 1, 0] = 1.0
            C_ova[i - 1, i] = -1.0

        print(f"   Specification matrix shape: {C_ova.shape}")

        result_ova = model.compute_bounds(method='CROWN', C=C_ova)

        print(f"\n   Results (class 0 vs others):")
        all_verified = True
        for i in range(1, output_size):
            lower = result_ova.lower()[i - 1]
            upper = result_ova.upper()[i - 1]

            if lower > 0:
                status = "✓ Class 0 wins"
                all_verified &= True
            else:
                status = "✗ Not verified"
                all_verified &= False

            print(f"   Class 0 - Class {i}: [{lower:.6f}, {upper:.6f}] {status}")

        if all_verified:
            print(f"\n   ✓✓✓ Verified: Class 0 is always the winner!")
        else:
            print(f"\n   Some comparisons could not be verified")

    # Example 4: Weighted combination
    print(f"\n5. Weighted output combination...")

    # Random weights
    np.random.seed(42)
    weights = np.random.randn(output_size).astype(np.float32)
    weights = weights / np.linalg.norm(weights)  # Normalize

    C_weighted = weights.reshape(1, -1)

    print(f"   Weights: {weights}")
    print(f"   Computing: {' + '.join([f'{w:.3f}*out[{i}]' for i, w in enumerate(weights)])}")

    result_weighted = model.compute_bounds(method='CROWN', C=C_weighted)

    print(f"\n   Weighted sum bounds: [{result_weighted.lower()[0]:.6f}, {result_weighted.upper()[0]:.6f}]")

    # Example 5: Multiple specifications at once
    print(f"\n6. Multiple specifications simultaneously...")

    # Combine several specifications
    C_multi = np.vstack([
        np.eye(output_size)[0],  # First output
        np.eye(output_size)[-1],  # Last output
        C_diff[0] if output_size >= 2 else np.eye(output_size)[0],  # First difference
    ]).astype(np.float32)

    print(f"   Specification matrix shape: {C_multi.shape}")
    print(f"   Computing 3 properties simultaneously")

    result_multi = model.compute_bounds(method='CROWN', C=C_multi)

    print(f"\n   Results:")
    print(f"   Property 1 (output[0]):         [{result_multi.lower()[0]:.6f}, {result_multi.upper()[0]:.6f}]")
    print(f"   Property 2 (output[-1]):        [{result_multi.lower()[1]:.6f}, {result_multi.upper()[1]:.6f}]")
    print(f"   Property 3 (out[0] - out[1]):   [{result_multi.lower()[2]:.6f}, {result_multi.upper()[2]:.6f}]")

    print("\n" + "="*60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
