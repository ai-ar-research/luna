#!/usr/bin/env python3
"""
Example 3: Alpha-CROWN Optimization

This example demonstrates:
- Using Alpha-CROWN for tighter bounds
- Configuring Alpha-CROWN parameters
- Comparing CROWN vs Alpha-CROWN
"""

from lunapy import TorchModel, LunaConfiguration
import numpy as np
import time
from pathlib import Path

def main():
    print("="*60)
    print("Example 3: Alpha-CROWN Optimization")
    print("="*60)

    repo_root = Path(__file__).parent.parent.parent
    onnx_path = str(repo_root / "resources" / "onnx" / "spec_test.onnx")
    vnnlib_path = str(repo_root / "resources" / "properties" / "spec_test.vnnlib")

    print(f"\n1. Loading model...")
    model = TorchModel(onnx_path, vnnlib_path)
    print(f"   Model: {model.getInputSize()} inputs → {model.getOutputSize()} outputs")

    # First, run CROWN as baseline
    print(f"\n2. Running CROWN (baseline)...")
    start = time.time()
    crown_result = model.compute_bounds(method='CROWN')
    crown_time = time.time() - start

    crown_lower = crown_result.lower()
    crown_upper = crown_result.upper()
    crown_width = np.sum(crown_upper - crown_lower)

    print(f"   Time: {crown_time:.4f}s")
    print(f"   Lower bounds: {crown_lower}")
    print(f"   Upper bounds: {crown_upper}")
    print(f"   Total width: {crown_width:.6f}")

    # Run Alpha-CROWN with default settings
    print(f"\n3. Running Alpha-CROWN (default settings)...")
    print(f"   Iterations: {LunaConfiguration.ALPHA_ITERATIONS}")
    print(f"   Learning rate: {LunaConfiguration.ALPHA_LR}")

    start = time.time()
    alpha_result = model.compute_bounds(method='alpha-CROWN')
    alpha_time = time.time() - start

    alpha_lower = alpha_result.lower()
    alpha_upper = alpha_result.upper()
    alpha_width = np.sum(alpha_upper - alpha_lower)

    print(f"   Time: {alpha_time:.4f}s")
    print(f"   Lower bounds: {alpha_lower}")
    print(f"   Upper bounds: {alpha_upper}")
    print(f"   Total width: {alpha_width:.6f}")

    # Compare
    print(f"\n4. Comparison:")
    print(f"   CROWN width:       {crown_width:.6f}")
    print(f"   Alpha-CROWN width: {alpha_width:.6f}")
    print(f"   Improvement:       {100 * (1 - alpha_width/crown_width):.2f}%")
    print(f"   Time overhead:     {alpha_time/crown_time:.2f}x")

    # Try with more iterations
    print(f"\n5. Running Alpha-CROWN with more iterations...")

    LunaConfiguration.ALPHA_ITERATIONS = 50
    LunaConfiguration.ALPHA_LR = 0.1
    LunaConfiguration.VERBOSE = False

    print(f"   Iterations: {LunaConfiguration.ALPHA_ITERATIONS}")
    print(f"   Learning rate: {LunaConfiguration.ALPHA_LR}")

    start = time.time()
    alpha_result2 = model.compute_bounds(method='alpha-CROWN')
    alpha_time2 = time.time() - start

    alpha_width2 = np.sum(alpha_result2.upper() - alpha_result2.lower())

    print(f"   Time: {alpha_time2:.4f}s")
    print(f"   Total width: {alpha_width2:.6f}")
    print(f"   Improvement over CROWN: {100 * (1 - alpha_width2/crown_width):.2f}%")

    # Per-output analysis
    print(f"\n6. Per-output bound tightness:")
    print(f"   {'Output':<10} {'CROWN Width':<15} {'Alpha Width':<15} {'Improvement':<15}")
    print(f"   {'-'*60}")

    for i in range(len(crown_lower)):
        crown_w = crown_upper[i] - crown_lower[i]
        alpha_w = alpha_result2.upper()[i] - alpha_result2.lower()[i]
        improvement = 100 * (1 - alpha_w/crown_w) if crown_w > 0 else 0

        print(f"   {i:<10} {crown_w:<15.6f} {alpha_w:<15.6f} {improvement:<15.2f}%")

    # Optimization settings exploration
    print(f"\n7. Optimizing only lower bounds...")

    LunaConfiguration.OPTIMIZE_LOWER = True
    LunaConfiguration.OPTIMIZE_UPPER = False
    LunaConfiguration.ALPHA_ITERATIONS = 20

    result_lower = model.compute_bounds(method='alpha-CROWN')

    print(f"   Lower bounds optimized: {result_lower.lower()}")
    print(f"   Upper bounds (CROWN):   {result_lower.upper()}")

    print("\n" + "="*60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
