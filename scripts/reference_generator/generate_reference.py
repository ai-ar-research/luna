#!/usr/bin/env python3
"""
Generate reference data from Python auto_LiRPA for cross-validation.

This script generates expected bounds from Python auto_LiRPA and saves them
as JSON files for comparison with the C++ implementation.
"""

import torch
import json
import numpy as np
import argparse
import sys
from pathlib import Path

try:
    from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
except ImportError:
    print("ERROR: auto_LiRPA not installed. Install with: pip install auto-LiRPA")
    sys.exit(1)

try:
    import onnx
    from onnx2pytorch import ConvertModel
except ImportError:
    print("ERROR: onnx or onnx2pytorch not installed.")
    print("Install with: pip install onnx onnx2pytorch")
    sys.exit(1)


def generate_reference_for_model(onnx_path, input_bounds, output_path,
                                  compute_alpha_crown=True, alpha_iterations=20):
    """
    Generate CROWN and AlphaCROWN bounds using auto_LiRPA.

    Args:
        onnx_path: Path to ONNX model file
        input_bounds: Dict with 'lower' and 'upper' lists, or path to VNN-LIB file
        output_path: Path to save JSON reference data
        compute_alpha_crown: Whether to compute AlphaCROWN bounds
        alpha_iterations: Number of AlphaCROWN optimization iterations
    """
    print(f"Loading ONNX model: {onnx_path}")
    onnx_model = onnx.load(onnx_path)
    model = ConvertModel(onnx_model)
    model.eval()

    # Parse input bounds
    if isinstance(input_bounds, dict):
        x_L = torch.tensor(input_bounds['lower'], dtype=torch.float32)
        x_U = torch.tensor(input_bounds['upper'], dtype=torch.float32)
    else:
        raise ValueError("input_bounds must be a dict with 'lower' and 'upper' keys")

    print(f"Input bounds shape: {x_L.shape}")
    print(f"Input lower range: [{x_L.min().item():.6f}, {x_L.max().item():.6f}]")
    print(f"Input upper range: [{x_U.min().item():.6f}, {x_U.max().item():.6f}]")

    # Create bounded model
    bounded_model = BoundedModule(model, x_L, device='cpu')
    ptb = PerturbationLpNorm(norm=np.inf, x_L=x_L, x_U=x_U)
    x = BoundedTensor((x_L + x_U) / 2, ptb)

    print("Computing CROWN bounds...")
    # CROWN bounds
    crown_lb, crown_ub = bounded_model.compute_bounds(x=(x,), method='CROWN')

    crown_lb_np = crown_lb.detach().cpu().numpy()
    crown_ub_np = crown_ub.detach().cpu().numpy()

    print(f"CROWN bounds computed: shape {crown_lb_np.shape}")
    print(f"CROWN lower range: [{crown_lb_np.min():.6f}, {crown_lb_np.max():.6f}]")
    print(f"CROWN upper range: [{crown_ub_np.min():.6f}, {crown_ub_np.max():.6f}]")

    alpha_crown_bounds = None
    if compute_alpha_crown:
        print(f"Computing AlphaCROWN bounds ({alpha_iterations} iterations)...")
        try:
            alpha_lb, alpha_ub = bounded_model.compute_bounds(
                x=(x,), method='alpha-CROWN',
                alpha=alpha_iterations)

            alpha_lb_np = alpha_lb.detach().cpu().numpy()
            alpha_ub_np = alpha_ub.detach().cpu().numpy()

            print(f"AlphaCROWN bounds computed: shape {alpha_lb_np.shape}")
            print(f"AlphaCROWN lower range: [{alpha_lb_np.min():.6f}, {alpha_lb_np.max():.6f}]")
            print(f"AlphaCROWN upper range: [{alpha_ub_np.min():.6f}, {alpha_ub_np.max():.6f}]")

            alpha_crown_bounds = {
                'lower': alpha_lb_np.tolist(),
                'upper': alpha_ub_np.tolist()
            }
        except Exception as e:
            print(f"Warning: AlphaCROWN computation failed: {e}")
            alpha_crown_bounds = None

    # Prepare reference data
    reference_data = {
        'model_path': str(onnx_path),
        'input_bounds': {
            'lower': x_L.numpy().tolist(),
            'upper': x_U.numpy().tolist()
        },
        'crown_bounds': {
            'lower': crown_lb_np.tolist(),
            'upper': crown_ub_np.tolist()
        },
        'tolerance': {
            'atol': 1e-3,
            'rtol': 1e-2
        }
    }

    if alpha_crown_bounds:
        reference_data['alpha_crown_bounds'] = alpha_crown_bounds

    # Save to JSON
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(reference_data, f, indent=2)

    print(f"Reference data saved to: {output_path}")
    return reference_data


def parse_vnnlib_bounds(vnnlib_path, input_size):
    """
    Parse input bounds from VNN-LIB file.
    Returns dict with 'lower' and 'upper' lists.
    """
    # Simplified VNN-LIB parser for input bounds
    # This is a basic implementation - full parser would handle all VNN-LIB syntax
    lower = [-float('inf')] * input_size
    upper = [float('inf')] * input_size

    with open(vnnlib_path, 'r') as f:
        content = f.read()

    # Simple parsing for (assert (>= X_i value)) and (assert (<= X_i value))
    import re

    # Find lower bounds: (>= X_i value)
    for match in re.finditer(r'\(>= X_(\d+) ([0-9.e\-]+)\)', content):
        idx = int(match.group(1))
        value = float(match.group(2))
        if idx < input_size:
            lower[idx] = max(lower[idx], value)

    # Find upper bounds: (<= X_i value)
    for match in re.finditer(r'\(<= X_(\d+) ([0-9.e\-]+)\)', content):
        idx = int(match.group(1))
        value = float(match.group(2))
        if idx < input_size:
            upper[idx] = min(upper[idx], value)

    return {'lower': lower, 'upper': upper}


def main():
    parser = argparse.ArgumentParser(
        description='Generate reference bounds from Python auto_LiRPA')
    parser.add_argument('--onnx', required=True, help='Path to ONNX model file')
    parser.add_argument('--vnnlib', help='Path to VNN-LIB file for input bounds')
    parser.add_argument('--input-size', type=int, help='Input size (required if using VNN-LIB)')
    parser.add_argument('--output', required=True, help='Output JSON file path')
    parser.add_argument('--no-alpha-crown', action='store_true',
                       help='Skip AlphaCROWN computation')
    parser.add_argument('--alpha-iterations', type=int, default=20,
                       help='Number of AlphaCROWN iterations')

    args = parser.parse_args()

    # Parse input bounds
    if args.vnnlib:
        if not args.input_size:
            print("ERROR: --input-size required when using --vnnlib")
            sys.exit(1)
        input_bounds = parse_vnnlib_bounds(args.vnnlib, args.input_size)
    else:
        # Default bounds (can be overridden)
        print("WARNING: No input bounds specified, using default [-1, 1]")
        input_bounds = {
            'lower': [-1.0] * args.input_size if args.input_size else [0.0] * 5,
            'upper': [1.0] * args.input_size if args.input_size else [1.0] * 5
        }

    try:
        generate_reference_for_model(
            args.onnx,
            input_bounds,
            args.output,
            compute_alpha_crown=not args.no_alpha_crown,
            alpha_iterations=args.alpha_iterations
        )
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
