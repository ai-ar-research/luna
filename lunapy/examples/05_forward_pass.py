#!/usr/bin/env python3
"""
Example 5: Forward Pass and Verification

This example demonstrates:
- Performing forward passes
- Verifying that concrete outputs are within computed bounds
- Sampling inputs within bound regions
"""

from lunapy import TorchModel
import numpy as np
from pathlib import Path

def main():
    print("="*60)
    print("Example 5: Forward Pass and Verification")
    print("="*60)

    repo_root = Path(__file__).parent.parent.parent
    onnx_path = str(repo_root / "resources" / "onnx" / "spec_test.onnx")

    print(f"\n1. Loading model and setting bounds...")
    model = TorchModel(onnx_path)

    # Set input bounds manually to avoid specification matrix from VNN-LIB
    input_size = model.getInputSize()
    lower = np.zeros(input_size, dtype=np.float32)
    upper = np.ones(input_size, dtype=np.float32)
    model.setInputBounds(lower, upper)

    print(f"   Input bounds: [{lower[0]:.3f}, {upper[0]:.3f}] (first dimension)")

    # Compute certified bounds
    print(f"\n2. Computing certified output bounds with CROWN...")
    certified_bounds = model.compute_bounds(method='CROWN')

    cert_lower = certified_bounds.lower()
    cert_upper = certified_bounds.upper()

    print(f"   Certified bounds:")
    for i, (l, u) in enumerate(zip(cert_lower, cert_upper)):
        print(f"     Output[{i}]: [{l:.6f}, {u:.6f}]")

    # Test points
    print(f"\n3. Testing concrete inputs...")

    # Test 1: Center point
    print(f"\n   Test 1: Center of input region")
    x_center = (lower + upper) / 2.0
    y_center = model.forward(x_center)

    print(f"   Output: {y_center}")
    print(f"   Checking if output is within certified bounds...")

    for i in range(len(y_center)):
        within_bounds = cert_lower[i] <= y_center[i] <= cert_upper[i]
        status = "✓" if within_bounds else "✗"
        print(f"     {status} Output[{i}]: {y_center[i]:.6f} in [{cert_lower[i]:.6f}, {cert_upper[i]:.6f}]")

    # Test 2: Corner points
    print(f"\n   Test 2: Corner points")

    corners = [
        lower,  # All lower bounds
        upper,  # All upper bounds
    ]

    for idx, corner in enumerate(corners):
        y_corner = model.forward(corner)
        print(f"\n   Corner {idx+1}: {'lower' if idx == 0 else 'upper'}")
        print(f"   Output: {y_corner}")

        all_within = True
        for i in range(len(y_corner)):
            within_bounds = cert_lower[i] - 1e-5 <= y_corner[i] <= cert_upper[i] + 1e-5
            all_within &= within_bounds

        print(f"   {'✓' if all_within else '✗'} All outputs within certified bounds")

    # Test 3: Random sampling
    print(f"\n   Test 3: Random sampling within input region")

    np.random.seed(42)
    num_samples = 10

    all_samples_valid = True
    violations = []

    for sample_idx in range(num_samples):
        # Random point in [lower, upper]
        alpha = np.random.random(len(lower)).astype(np.float32)
        x_sample = lower + alpha * (upper - lower)

        y_sample = model.forward(x_sample)

        # Check if within bounds
        sample_valid = True
        for i in range(len(y_sample)):
            # Allow small numerical tolerance
            tolerance = 1e-4
            within = (cert_lower[i] - tolerance <= y_sample[i] <= cert_upper[i] + tolerance)

            if not within:
                sample_valid = False
                violations.append({
                    'sample': sample_idx,
                    'output': i,
                    'value': y_sample[i],
                    'bounds': (cert_lower[i], cert_upper[i])
                })

        all_samples_valid &= sample_valid

    print(f"   Tested {num_samples} random samples")

    if all_samples_valid:
        print(f"   ✓✓✓ All {num_samples} samples within certified bounds!")
    else:
        print(f"   ✗ Found {len(violations)} violations (may be due to loose bounds)")
        for v in violations[:3]:  # Show first 3
            print(f"     Sample {v['sample']}, Output {v['output']}: "
                  f"{v['value']:.6f} not in [{v['bounds'][0]:.6f}, {v['bounds'][1]:.6f}]")

    # Test 4: Adversarial direction (worst case)
    print(f"\n   Test 4: Exploring bound tightness")

    # Try to maximize first output
    print(f"   Attempting to maximize output[0]...")

    best_output = -np.inf
    best_input = None

    for _ in range(100):
        x_random = lower + np.random.random(len(lower)).astype(np.float32) * (upper - lower)
        y_random = model.forward(x_random)

        if y_random[0] > best_output:
            best_output = y_random[0]
            best_input = x_random

    print(f"   Best output[0] found: {best_output:.6f}")
    print(f"   Certified upper bound: {cert_upper[0]:.6f}")
    print(f"   Gap: {cert_upper[0] - best_output:.6f} ({100*(cert_upper[0] - best_output)/cert_upper[0]:.2f}%)")

    # Test 5: Batched forward passes
    print(f"\n4. Performance: Single vs multiple forward passes")

    import time

    # Single pass
    start = time.time()
    for _ in range(100):
        x = lower + np.random.random(len(lower)).astype(np.float32) * (upper - lower)
        y = model.forward(x)
    single_time = time.time() - start

    print(f"   100 forward passes: {single_time:.4f}s ({single_time/100*1000:.2f}ms per pass)")

    print("\n" + "="*60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
