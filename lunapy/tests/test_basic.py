"""
Basic tests for lunapy core functionality
"""

import pytest
import numpy as np
from lunapy import TorchModel, BoundedTensor, LunaConfiguration


class TestBoundedTensor:
    """Tests for BoundedTensor class"""

    def test_create_bounded_tensor(self):
        """Test creating a BoundedTensor"""
        lower = np.array([0.0, 0.0], dtype=np.float32)
        upper = np.array([1.0, 1.0], dtype=np.float32)

        bt = BoundedTensor(lower, upper)

        assert bt is not None
        assert bt.lower().shape == (2,)
        assert bt.upper().shape == (2,)
        np.testing.assert_array_equal(bt.lower(), lower)
        np.testing.assert_array_equal(bt.upper(), upper)

    def test_bounded_tensor_from_lists(self):
        """Test creating BoundedTensor from Python lists"""
        lower = [0.0, 0.0]
        upper = [1.0, 1.0]

        bt = BoundedTensor(lower, upper)

        assert bt.lower().shape == (2,)
        assert bt.upper().shape == (2,)

    def test_bounded_tensor_shapes(self):
        """Test BoundedTensor with different shapes"""
        # 1D
        bt1 = BoundedTensor(np.zeros(5), np.ones(5))
        assert bt1.lower().shape == (5,)

        # 2D
        bt2 = BoundedTensor(np.zeros((3, 4)), np.ones((3, 4)))
        assert bt2.lower().shape == (3, 4)

        # 3D
        bt3 = BoundedTensor(np.zeros((2, 3, 4)), np.ones((2, 3, 4)))
        assert bt3.lower().shape == (2, 3, 4)


class TestTorchModelBasic:
    """Basic tests for TorchModel class"""

    def test_import(self):
        """Test that we can import TorchModel"""
        assert TorchModel is not None

    def test_create_model_onnx_only(self, spec_test_onnx, check_resources):
        """Test creating TorchModel with only ONNX file"""
        model = TorchModel(spec_test_onnx)

        assert model is not None
        assert model.getInputSize() > 0
        assert model.getOutputSize() > 0
        assert model.getNumNodes() > 0

    def test_create_model_with_vnnlib(self, spec_test_onnx, spec_test_vnnlib, check_resources):
        """Test creating TorchModel with ONNX and VNN-LIB"""
        model = TorchModel(spec_test_onnx, spec_test_vnnlib)

        assert model is not None
        assert model.hasInputBounds()

    def test_model_repr(self, spec_test_onnx, check_resources):
        """Test model string representation"""
        model = TorchModel(spec_test_onnx)
        repr_str = repr(model)

        assert "TorchModel" in repr_str
        assert "nodes=" in repr_str
        assert "input_size=" in repr_str
        assert "output_size=" in repr_str


class TestInputBounds:
    """Tests for input bound management"""

    def test_set_input_bounds_numpy(self, spec_test_onnx, check_resources):
        """Test setting input bounds from NumPy arrays"""
        model = TorchModel(spec_test_onnx)

        input_size = model.getInputSize()
        lower = np.zeros(input_size, dtype=np.float32)
        upper = np.ones(input_size, dtype=np.float32)

        model.setInputBounds(lower, upper)

        assert model.hasInputBounds()
        bounds = model.getInputBounds()
        assert bounds.lower().shape[0] == input_size
        assert bounds.upper().shape[0] == input_size

    def test_set_input_bounds_lists(self, spec_test_onnx, check_resources):
        """Test setting input bounds from Python lists"""
        model = TorchModel(spec_test_onnx)

        input_size = model.getInputSize()
        lower = [0.0] * input_size
        upper = [1.0] * input_size

        model.setInputBounds(lower, upper)

        assert model.hasInputBounds()

    def test_get_input_bounds(self, spec_test_onnx, spec_test_vnnlib, check_resources):
        """Test getting input bounds from VNN-LIB"""
        model = TorchModel(spec_test_onnx, spec_test_vnnlib)

        assert model.hasInputBounds()
        bounds = model.getInputBounds()

        assert bounds.lower() is not None
        assert bounds.upper() is not None
        assert len(bounds.lower()) == model.getInputSize()


class TestConfiguration:
    """Tests for LunaConfiguration"""

    def test_configuration_exists(self):
        """Test that LunaConfiguration is accessible"""
        assert LunaConfiguration is not None

    def test_set_verbose(self):
        """Test setting VERBOSE flag"""
        original = LunaConfiguration.VERBOSE

        LunaConfiguration.VERBOSE = True
        assert LunaConfiguration.VERBOSE == True

        LunaConfiguration.VERBOSE = False
        assert LunaConfiguration.VERBOSE == False

        # Restore
        LunaConfiguration.VERBOSE = original

    def test_set_alpha_iterations(self):
        """Test setting ALPHA_ITERATIONS"""
        original = LunaConfiguration.ALPHA_ITERATIONS

        LunaConfiguration.ALPHA_ITERATIONS = 50
        assert LunaConfiguration.ALPHA_ITERATIONS == 50

        # Restore
        LunaConfiguration.ALPHA_ITERATIONS = original

    def test_set_alpha_lr(self):
        """Test setting ALPHA_LR"""
        original = LunaConfiguration.ALPHA_LR

        LunaConfiguration.ALPHA_LR = 0.05
        assert abs(LunaConfiguration.ALPHA_LR - 0.05) < 1e-6

        # Restore
        LunaConfiguration.ALPHA_LR = original


class TestErrorHandling:
    """Tests for error handling"""

    def test_nonexistent_onnx_file(self):
        """Test error when ONNX file doesn't exist"""
        with pytest.raises(Exception):
            TorchModel("nonexistent_file.onnx")

    def test_compute_bounds_without_input_bounds(self, spec_test_onnx, check_resources):
        """Test error when computing bounds without setting input bounds"""
        model = TorchModel(spec_test_onnx)

        # Should raise error because no input bounds set
        with pytest.raises(Exception):
            model.compute_bounds()

    def test_invalid_method_name(self, spec_test_onnx, spec_test_vnnlib, check_resources):
        """Test error with invalid analysis method"""
        model = TorchModel(spec_test_onnx, spec_test_vnnlib)

        with pytest.raises(Exception):
            model.compute_bounds(method='INVALID_METHOD')
