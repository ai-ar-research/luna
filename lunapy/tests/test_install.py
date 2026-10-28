"""
Simple installation verification test

This test can be run immediately after installation to verify
that lunapy is working correctly.
"""

import pytest


def test_import_lunapy():
    """Test that lunapy can be imported"""
    try:
        import lunapy
        assert lunapy is not None
    except ImportError as e:
        pytest.fail(f"Failed to import lunapy: {e}")


def test_import_core_components():
    """Test importing core components"""
    from lunapy import TorchModel, BoundedTensor, LunaConfiguration

    assert TorchModel is not None
    assert BoundedTensor is not None
    assert LunaConfiguration is not None


def test_create_bounded_tensor():
    """Test creating a simple BoundedTensor"""
    import numpy as np
    from lunapy import BoundedTensor

    lower = np.array([0.0, 0.0], dtype=np.float32)
    upper = np.array([1.0, 1.0], dtype=np.float32)

    bt = BoundedTensor(lower, upper)

    assert bt is not None
    assert bt.lower() is not None
    assert bt.upper() is not None


def test_version():
    """Test that version is accessible"""
    import lunapy

    assert hasattr(lunapy, '__version__')
    assert isinstance(lunapy.__version__, str)


def test_configuration_accessible():
    """Test that LunaConfiguration is accessible"""
    from lunapy import LunaConfiguration

    # Should be able to read configuration values
    assert hasattr(LunaConfiguration, 'VERBOSE')
    assert hasattr(LunaConfiguration, 'ALPHA_ITERATIONS')
    assert hasattr(LunaConfiguration, 'ALPHA_LR')


if __name__ == '__main__':
    """Run installation tests"""
    print("Running lunapy installation tests...")

    try:
        test_import_lunapy()
        print("✓ Import test passed")

        test_import_core_components()
        print("✓ Core components import test passed")

        test_create_bounded_tensor()
        print("✓ BoundedTensor creation test passed")

        test_version()
        print("✓ Version test passed")

        test_configuration_accessible()
        print("✓ Configuration test passed")

        print("\n" + "="*50)
        print("All installation tests passed!")
        print("lunapy is installed correctly.")
        print("="*50)

    except Exception as e:
        print(f"\n✗ Installation test failed: {e}")
        print("\nPlease ensure lunapy is built correctly:")
        print("  pip install -e .")
        print("or")
        print("  python setup.py build_ext --inplace")
        raise
