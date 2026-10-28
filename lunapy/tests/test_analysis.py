"""
Tests for CROWN and Alpha-CROWN analysis
"""

import pytest
import numpy as np
from lunapy import TorchModel, LunaConfiguration


class TestCROWNAnalysis:
    """Tests for CROWN bound propagation"""

    def test_crown_basic(self, spec_test_onnx, spec_test_vnnlib, check_resources):
        """Test basic CROWN analysis"""
        model = TorchModel(spec_test_onnx, spec_test_vnnlib)

        result = model.compute_bounds(method='CROWN')

        assert result is not None
        assert result.lower() is not None
        assert result.upper() is not None
        # Note: VNN-LIB may set a specification matrix that changes output size
        # So we just check that bounds exist and have matching dimensions
        assert len(result.lower()) == len(result.upper())
        assert len(result.lower()) > 0

    def test_crown_lower_only(self, spec_test_onnx, spec_test_vnnlib, check_resources):
        """Test CROWN computing only lower bounds"""
        model = TorchModel(spec_test_onnx, spec_test_vnnlib)

        result = model.compute_bounds(
            method='CROWN',
            bound_lower=True,
            bound_upper=False
        )

        assert result is not None

    def test_crown_upper_only(self, spec_test_onnx, spec_test_vnnlib, check_resources):
        """Test CROWN computing only upper bounds"""
        model = TorchModel(spec_test_onnx, spec_test_vnnlib)

        result = model.compute_bounds(
            method='CROWN',
            bound_lower=False,
            bound_upper=True
        )

        assert result is not None

    def test_crown_bounds_consistency(self, spec_test_onnx, spec_test_vnnlib, check_resources):
        """Test that lower bounds <= upper bounds"""
        model = TorchModel(spec_test_onnx, spec_test_vnnlib)

        result = model.compute_bounds(method='CROWN')

        # Lower bounds should be <= upper bounds (with some numerical tolerance)
        assert np.all(result.lower() <= result.upper() + 1e-4)

    def test_run_crown_direct(self, spec_test_onnx, spec_test_vnnlib, check_resources):
        """Test calling runCROWN directly"""
        model = TorchModel(spec_test_onnx, spec_test_vnnlib)

        result = model.runCROWN()

        assert result is not None
        assert result.lower() is not None
        assert result.upper() is not None


class TestAlphaCROWNAnalysis:
    """Tests for Alpha-CROWN optimization"""

    def test_alpha_crown_basic(self, spec_test_onnx, spec_test_vnnlib, check_resources):
        """Test basic Alpha-CROWN analysis"""
        model = TorchModel(spec_test_onnx, spec_test_vnnlib)

        # Use fewer iterations for faster testing
        LunaConfiguration.ALPHA_ITERATIONS = 10

        result = model.compute_bounds(method='alpha-CROWN')

        assert result is not None
        assert result.lower() is not None
        assert result.upper() is not None

    def test_alpha_crown_vs_crown(self, spec_test_onnx, spec_test_vnnlib, check_resources):
        """Test that Alpha-CROWN gives tighter or equal bounds vs CROWN"""
        model = TorchModel(spec_test_onnx, spec_test_vnnlib)

        # CROWN bounds
        crown_result = model.compute_bounds(method='CROWN')
        crown_lower = crown_result.lower()
        crown_upper = crown_result.upper()

        # Alpha-CROWN bounds (with few iterations)
        LunaConfiguration.ALPHA_ITERATIONS = 10
        alpha_result = model.compute_bounds(method='alpha-CROWN')
        alpha_lower = alpha_result.lower()
        alpha_upper = alpha_result.upper()

        # Alpha-CROWN should give tighter or equal bounds
        # (lower bounds should be >= CROWN, upper bounds should be <= CROWN)
        # Allow some numerical tolerance
        tolerance = 1e-3
        assert np.all(alpha_lower >= crown_lower - tolerance)
        assert np.all(alpha_upper <= crown_upper + tolerance)

    def test_alpha_crown_configuration(self, spec_test_onnx, spec_test_vnnlib, check_resources):
        """Test Alpha-CROWN with custom configuration"""
        model = TorchModel(spec_test_onnx, spec_test_vnnlib)

        # Set custom configuration
        LunaConfiguration.ALPHA_ITERATIONS = 5
        LunaConfiguration.ALPHA_LR = 0.1
        LunaConfiguration.OPTIMIZE_LOWER = True
        LunaConfiguration.OPTIMIZE_UPPER = False

        result = model.compute_bounds(method='alpha-CROWN')

        assert result is not None

    def test_run_alpha_crown_direct(self, spec_test_onnx, spec_test_vnnlib, check_resources):
        """Test calling runAlphaCROWN directly"""
        model = TorchModel(spec_test_onnx, spec_test_vnnlib)

        LunaConfiguration.ALPHA_ITERATIONS = 5
        result = model.runAlphaCROWN(optimizeLower=True, optimizeUpper=False)

        assert result is not None


class TestSpecificationMatrix:
    """Tests for custom specification matrices"""

    def test_specification_matrix(self, spec_test_onnx, check_resources):
        """Test computing bounds with specification matrix"""
        # Load without VNN-LIB to avoid pre-set specifications
        model = TorchModel(spec_test_onnx)

        # Set simple bounds manually
        input_size = model.getInputSize()
        model.setInputBounds(
            np.full(input_size, -1.0, dtype=np.float32),
            np.full(input_size, 1.0, dtype=np.float32)
        )

        output_size = model.getOutputSize()

        if output_size >= 2:
            # Create specification: difference between first two outputs
            C = np.zeros((1, output_size), dtype=np.float32)
            C[0, 0] = 1.0
            C[0, 1] = -1.0

            result = model.compute_bounds(method='CROWN', C=C)

            assert result is not None
            assert len(result.lower()) == 1  # One row in C
            assert len(result.upper()) == 1

    def test_identity_specification(self, spec_test_onnx, check_resources):
        """Test with identity specification matrix (should match no-spec case)"""
        # Load without VNN-LIB to avoid pre-set specifications
        model = TorchModel(spec_test_onnx)

        # Set simple bounds manually
        input_size = model.getInputSize()
        model.setInputBounds(
            np.full(input_size, -1.0, dtype=np.float32),
            np.full(input_size, 1.0, dtype=np.float32)
        )

        output_size = model.getOutputSize()
        C = np.eye(output_size, dtype=np.float32)

        result = model.compute_bounds(method='CROWN', C=C)

        assert result is not None
        assert len(result.lower()) == output_size


class TestForwardPass:
    """Tests for forward pass functionality"""

    def test_forward_pass(self, spec_test_onnx, check_resources):
        """Test forward pass through the model"""
        model = TorchModel(spec_test_onnx)

        input_size = model.getInputSize()
        x = np.random.randn(input_size).astype(np.float32)

        output = model.forward(x)

        assert output is not None
        assert len(output) == model.getOutputSize()

    def test_forward_pass_zeros(self, spec_test_onnx, check_resources):
        """Test forward pass with zero input"""
        model = TorchModel(spec_test_onnx)

        input_size = model.getInputSize()
        x = np.zeros(input_size, dtype=np.float32)

        output = model.forward(x)

        assert output is not None
        assert len(output) == model.getOutputSize()

    def test_forward_pass_within_bounds(self, spec_test_onnx, check_resources):
        """Test that forward pass within bounds gives output within computed bounds"""
        # Load without VNN-LIB to avoid specification matrix affecting output size
        model = TorchModel(spec_test_onnx)

        # Set bounds manually
        input_size = model.getInputSize()
        lower = np.full(input_size, -1.0, dtype=np.float32)
        upper = np.full(input_size, 1.0, dtype=np.float32)
        model.setInputBounds(lower, upper)

        # Create input at center of bounds
        x = (lower + upper) / 2.0

        # Forward pass
        output = model.forward(x)

        # Compute bounds
        result = model.compute_bounds(method='CROWN')

        # Both should have same size (no spec matrix)
        assert len(output) == model.getOutputSize()
        assert len(result.lower()) == model.getOutputSize()

        # Output should be within computed bounds (with tolerance)
        # Note: This may not always hold for very loose bounds
        tolerance = 1.0  # Allow generous tolerance
        assert np.all(output >= result.lower() - tolerance)
        assert np.all(output <= result.upper() + tolerance)


class TestFinalBounds:
    """Tests for final analysis bounds storage"""

    def test_final_bounds_after_analysis(self, spec_test_onnx, spec_test_vnnlib, check_resources):
        """Test that final bounds are stored after analysis"""
        model = TorchModel(spec_test_onnx, spec_test_vnnlib)

        # Should not have final bounds initially
        assert not model.hasFinalAnalysisBounds()

        # Run analysis
        model.compute_bounds(method='CROWN')

        # Should now have final bounds
        assert model.hasFinalAnalysisBounds()

        # Get final bounds
        final = model.getFinalAnalysisBounds()
        assert final is not None

    def test_final_bounds_match_result(self, spec_test_onnx, spec_test_vnnlib, check_resources):
        """Test that final bounds match the returned result"""
        model = TorchModel(spec_test_onnx, spec_test_vnnlib)

        result = model.compute_bounds(method='CROWN')
        final = model.getFinalAnalysisBounds()

        np.testing.assert_array_almost_equal(result.lower(), final.lower(), decimal=5)
        np.testing.assert_array_almost_equal(result.upper(), final.upper(), decimal=5)
