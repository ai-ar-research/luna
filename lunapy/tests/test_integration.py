"""
Integration tests for complete workflows
"""

import pytest
import numpy as np
from lunapy import TorchModel, LunaConfiguration


class TestCompleteWorkflow:
    """Test complete analysis workflows"""

    def test_workflow_onnx_vnnlib_crown(self, spec_test_onnx, spec_test_vnnlib, check_resources):
        """Test complete workflow: load ONNX+VNN-LIB, run CROWN"""
        # Step 1: Load model
        model = TorchModel(spec_test_onnx, spec_test_vnnlib)

        # Step 2: Verify bounds loaded
        assert model.hasInputBounds()

        # Step 3: Run CROWN
        bounds = model.compute_bounds(method='CROWN')

        # Step 4: Verify results
        assert bounds is not None
        assert np.all(bounds.lower() <= bounds.upper())

        # Step 5: Check final bounds stored
        assert model.hasFinalAnalysisBounds()

    def test_workflow_manual_bounds(self, spec_test_onnx, check_resources):
        """Test workflow with manually set bounds"""
        # Step 1: Load model
        model = TorchModel(spec_test_onnx)

        # Step 2: Set bounds manually
        input_size = model.getInputSize()
        model.setInputBounds(
            lower=np.full(input_size, -1.0, dtype=np.float32),
            upper=np.full(input_size, 1.0, dtype=np.float32)
        )

        # Step 3: Run analysis
        bounds = model.compute_bounds(method='CROWN')

        # Step 4: Verify
        assert bounds is not None

    def test_workflow_compare_methods(self, spec_test_onnx, spec_test_vnnlib, check_resources):
        """Test comparing CROWN and Alpha-CROWN"""
        model = TorchModel(spec_test_onnx, spec_test_vnnlib)

        # Run CROWN
        crown_bounds = model.compute_bounds(method='CROWN')
        crown_width = np.sum(crown_bounds.upper() - crown_bounds.lower())

        # Run Alpha-CROWN
        LunaConfiguration.ALPHA_ITERATIONS = 10
        alpha_bounds = model.compute_bounds(method='alpha-CROWN')
        alpha_width = np.sum(alpha_bounds.upper() - alpha_bounds.lower())

        # Alpha-CROWN should have tighter or equal total width
        assert alpha_width <= crown_width + 1e-2  # Small tolerance

    def test_workflow_with_specification(self, spec_test_onnx, check_resources):
        """Test workflow with custom specification"""
        # Load without VNN-LIB to avoid pre-set specifications
        model = TorchModel(spec_test_onnx)

        # Set bounds manually
        input_size = model.getInputSize()
        model.setInputBounds(
            np.full(input_size, -1.0, dtype=np.float32),
            np.full(input_size, 1.0, dtype=np.float32)
        )

        output_size = model.getOutputSize()
        if output_size < 2:
            pytest.skip("Need at least 2 outputs for this test")

        # Create specification matrix
        C = np.zeros((output_size - 1, output_size), dtype=np.float32)
        for i in range(output_size - 1):
            C[i, i] = 1.0
            C[i, i + 1] = -1.0

        # Run analysis with specification
        bounds = model.compute_bounds(method='CROWN', C=C)
        assert len(bounds.lower()) == output_size - 1

    def test_workflow_forward_then_bounds(self, spec_test_onnx, spec_test_vnnlib, check_resources):
        """Test forward pass followed by bound computation"""
        model = TorchModel(spec_test_onnx, spec_test_vnnlib)

        # Get input bounds
        input_bounds = model.getInputBounds()

        # Forward pass at center
        x = (input_bounds.lower() + input_bounds.upper()) / 2.0
        output = model.forward(x)

        # Compute bounds
        bounds = model.compute_bounds(method='CROWN')

        # Both should succeed
        assert output is not None
        assert bounds is not None


class TestMultipleModels:
    """Test using multiple models simultaneously"""

    def test_two_models_different_files(self, spec_test_onnx, spec_test_vnnlib, check_resources):
        """Test creating two models from same file"""
        model1 = TorchModel(spec_test_onnx, spec_test_vnnlib)
        model2 = TorchModel(spec_test_onnx, spec_test_vnnlib)

        # Both should work independently
        bounds1 = model1.compute_bounds(method='CROWN')
        bounds2 = model2.compute_bounds(method='CROWN')

        # Results should be identical
        np.testing.assert_array_almost_equal(
            bounds1.lower(), bounds2.lower(), decimal=5
        )
        np.testing.assert_array_almost_equal(
            bounds1.upper(), bounds2.upper(), decimal=5
        )


class TestRobustness:
    """Test robustness and edge cases"""

    def test_very_tight_bounds(self, spec_test_onnx, check_resources):
        """Test with very tight input bounds"""
        model = TorchModel(spec_test_onnx)

        input_size = model.getInputSize()
        center = np.zeros(input_size, dtype=np.float32)
        epsilon = 0.001

        model.setInputBounds(
            lower=center - epsilon,
            upper=center + epsilon
        )

        bounds = model.compute_bounds(method='CROWN')

        # Output bounds should also be tight
        output_width = np.max(bounds.upper() - bounds.lower())
        assert output_width < 10.0  # Reasonable bound

    def test_asymmetric_bounds(self, spec_test_onnx, check_resources):
        """Test with asymmetric input bounds"""
        model = TorchModel(spec_test_onnx)

        input_size = model.getInputSize()
        lower = np.random.uniform(-2, -0.5, input_size).astype(np.float32)
        upper = np.random.uniform(0.5, 2, input_size).astype(np.float32)

        model.setInputBounds(lower, upper)
        bounds = model.compute_bounds(method='CROWN')

        assert bounds is not None
        assert np.all(bounds.lower() <= bounds.upper())
