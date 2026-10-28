# lunapy Tests

Comprehensive test suite for the lunapy Python bindings.

## Test Structure

```
tests/
├── conftest.py              # Pytest fixtures and configuration
├── test_install.py          # Installation verification tests
├── test_basic.py            # Basic functionality tests
├── test_analysis.py         # CROWN and Alpha-CROWN tests
└── test_integration.py      # Integration and workflow tests
```

## Running Tests

### Run All Tests

```bash
# From lunapy directory
pytest

# Or from repository root
pytest lunapy/tests/

# With verbose output
pytest -v

# With coverage
pytest --cov=lunapy --cov-report=html
```

### Run Specific Test Files

```bash
pytest tests/test_basic.py
pytest tests/test_analysis.py
pytest tests/test_integration.py
```

### Run Specific Tests

```bash
# By test name
pytest tests/test_basic.py::TestBoundedTensor::test_create_bounded_tensor

# By keyword
pytest -k "crown"  # Run all tests with "crown" in the name
pytest -k "alpha"  # Run all Alpha-CROWN tests
```

### Quick Installation Check

```bash
# Just verify installation works
python tests/test_install.py

# Or with pytest
pytest tests/test_install.py -v
```

## Test Categories

### Installation Tests (`test_install.py`)
- Import verification
- Core component availability
- Basic object creation
- **Run this first** after installation

### Basic Tests (`test_basic.py`)
- `TestBoundedTensor`: BoundedTensor creation and manipulation
- `TestTorchModelBasic`: TorchModel creation and properties
- `TestInputBounds`: Input bound management
- `TestConfiguration`: LunaConfiguration settings
- `TestErrorHandling`: Error cases and exceptions

### Analysis Tests (`test_analysis.py`)
- `TestCROWNAnalysis`: CROWN bound propagation
- `TestAlphaCROWNAnalysis`: Alpha-CROWN optimization
- `TestSpecificationMatrix`: Custom specification matrices
- `TestForwardPass`: Forward pass functionality
- `TestFinalBounds`: Final bounds storage

### Integration Tests (`test_integration.py`)
- `TestCompleteWorkflow`: End-to-end workflows
- `TestMultipleModels`: Multiple model instances
- `TestRobustness`: Edge cases and robustness

## Test Requirements

### Required Files
Tests expect the following resources to exist:
```
resources/onnx/spec_test.onnx
resources/properties/spec_test.vnnlib
```

If these files are not found, tests will be skipped automatically.

### Dependencies
```bash
pip install pytest pytest-cov numpy
```

## Test Fixtures

Common fixtures available in all tests (defined in `conftest.py`):

- `spec_test_onnx`: Path to spec_test.onnx
- `spec_test_vnnlib`: Path to spec_test.vnnlib
- `simple_input_bounds`: Dictionary with simple lower/upper bounds
- `check_resources`: Skips test if resources not found

## Writing New Tests

Example test structure:

```python
import pytest
from lunapy import TorchModel

class TestMyFeature:
    """Tests for my new feature"""

    def test_basic_functionality(self, spec_test_onnx, check_resources):
        """Test basic usage of feature"""
        model = TorchModel(spec_test_onnx)

        # Your test code here
        assert model is not None

    def test_error_handling(self):
        """Test error cases"""
        with pytest.raises(Exception):
            # Code that should raise exception
            pass
```

## Continuous Integration

Tests are designed to work in CI environments:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    pip install pytest pytest-cov
    pytest lunapy/tests/ -v
```

## Test Coverage

Generate coverage report:

```bash
pytest --cov=lunapy --cov-report=html --cov-report=term
```

View HTML report:
```bash
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

## Debugging Tests

### Run with detailed output
```bash
pytest -vv -s  # -s shows print statements
```

### Run specific test with debugger
```bash
pytest tests/test_basic.py::test_name --pdb
```

### Show test durations
```bash
pytest --durations=10  # Show 10 slowest tests
```

## Expected Test Results

With all resources available:
- ✓ 50+ tests should pass
- ⊘ Some tests may be skipped if resources unavailable
- ✗ No tests should fail (if installation is correct)

## Troubleshooting

**Import errors**: Ensure lunapy is installed
```bash
pip install -e .
```

**Resource not found**: Tests will skip automatically, or provide correct paths in conftest.py

**Segmentation fault**: Check that C++ extension is built correctly

**Random failures**: Some numerical tests have tolerance; may rarely fail due to precision

## Contributing

When adding new features:
1. Add corresponding tests
2. Ensure all existing tests pass
3. Maintain >80% code coverage
4. Document new test fixtures
