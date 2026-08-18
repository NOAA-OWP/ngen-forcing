# Tests README

This directory contains tests for the NextGen Forcing BMI Engine.

## Initial test data

Tests data is included in the `test_data` directory and includes configs, gpkgs, esmf_meshes, expected results and actual results. While the configs, gpkgs, esmf_meshes and expectd results are included in the repo and can be used as is, the following steps can be taken to re-create these test inputs.

---
The initial test data was generated using `nwm-rte` to create a calibration realization
for gage 01123000, starting at time 2013-07-01 00:00:00, and running for 3 timesteps,
using `nwm-rte's` run_suite.sh.  See RETRO_FORCING_CONFIG_FILE__AORC_CONUS.

More specifically the initial expected test data was developed with these specific configurations in `config.bashrc`.
```
REPO_TAG_FCST_MGR="856fc0e1201076df909e56c7cd384f58e82965a2"
REPO_TAG_MSW_MGR="693c206a22b5e9ffcca3103166c0ca59e2b11b25"
REPO_TAG_CAL_MGR="7e56bf01477ea77e72dfb25a166ac26ff6090ecb"
REPO_TAG_NGEN_FORCING="LOCAL"
NGEN_SOURCE_MODE="ghcr"
NGEN_BASE__REMOTE_GHCR_TAG="844c5f6"
```

And these two commands in `nwm-rte's` `run_suite.sh`:
```bash
docker_run python "/ngen-app/bin/bin_mounted/run_calibration.py" -n 2 -fsrc "aorc" -start "2013-07-01 00:00:00" -dur 3

docker_run python "/ngen-app/bin/bin_mounted/run_forecast.py" -fconfig "short_range" -dt "2025-07-10 04:00:00" -rname "fcst_run1_short_range"
```
## Test Structure

The test suite is organized into the following modules:

- **`esmf_regrid/`** - Tests for ESMF regridding functionality
- **`geomod/`** - Tests for geomod components
- **`input_forcing/`** - Tests for input forcing data processing
- **`bmi_model/`** - Tests for the BMI model lifecycle
- **`test_utils.py`** - Shared test utilities and fixtures
- **`conftest.py`** - Pytest configuration and shared fixtures

## Prerequisites
### Setup requirements:
    1. Create the forcing config.yml files using RTE.
    2. Enter the RTE devcontainer.

### Required Dependencies

The test suite requires Python 3.11 or higher. Install the package with test dependencies inside of the `dev container`:

```bash
# From the repository root directory
pip install -e ".[develop]"
```

Or install pytest directly inside of the `dev container`:

```bash
pip install pytest
```

### Additional Requirements

Ensure all main package dependencies are installed inside of the `dev container` (this typically should happen when the `dev container` is built):

```bash
pip install -e .
```

## Running Tests

### Run All Tests From the Dev Container

```bash
Single processor: (cd src/ngen-forcing && pytest )
Multiple processors: ( cd src/ngen-forcing && mpirun -n 2 pytest )
```
### Run Specific Test Modules From the Dev Container

Run tests for a specific module:

```bash
# ESMF regridding tests
Single processor: ( cd src/ngen-forcing && pytest tests/esmf_regrid)
Multiple processors: ( cd src/ngen-forcing && mpirun -n 2 pytest tests/esmf_regrid)

# GeoMod tests
Single processor: ( cd src/ngen-forcing && pytest tests/geomod)
Multiple processors: ( cd src/ngen-forcing && mpirun -n 2 pytest tests/geomod)

# Input forcing tests
Single processor: ( cd src/ngen-forcing && pytest tests/input_forcing)
Multiple processors: ( cd src/ngen-forcing && mpirun -n 2 pytest tests/input_forcing)

# BMI model tests
Single processor: ( cd src/ngen-forcing && pytest tests/bmi_model)
Multiple processors: ( cd src/ngen-forcing && mpirun -n 2 pytest tests/bmi_model)
```

Create new test output data (creates expected outputs for subsequent tests)
```bash
# ESMF regridding tests
Single processor: ( cd src/ngen-forcing && FORCING_PYTEST_WRITE_TEST_EXPECTED_DATA=true pytest tests/esmf_regrid)
Multiple processors: ( cd src/ngen-forcing && FORCING_PYTEST_WRITE_TEST_EXPECTED_DATA=true mpirun -n 2 pytest tests/esmf_regrid)

# GeoMod tests
Single processor: ( cd src/ngen-forcing && FORCING_PYTEST_WRITE_TEST_EXPECTED_DATA=true pytest tests/geomod)
Multiple processors: ( cd src/ngen-forcing && FORCING_PYTEST_WRITE_TEST_EXPECTED_DATA=true mpirun -n 2 pytest tests/geomod)

# Input forcing tests
Single processor: ( cd src/ngen-forcing && FORCING_PYTEST_WRITE_TEST_EXPECTED_DATA=true pytest tests/input_forcing)
Multiple processors: ( cd src/ngen-forcing && FORCING_PYTEST_WRITE_TEST_EXPECTED_DATA=true mpirun -n 2 pytest tests/input_forcing)

# BMI model tests
Single processor: ( cd src/ngen-forcing && FORCING_PYTEST_WRITE_TEST_EXPECTED_DATA=true pytest tests/bmi_model)
Multiple processors: ( cd src/ngen-forcing && FORCING_PYTEST_WRITE_TEST_EXPECTED_DATA=true mpirun -n 2 pytest tests/bmi_model)
```

In the rare case where you want to create new `expected` data and run the tests using `old` variable names use the following for `Input Forcing Tests`:
```bash
# Input forcing tests
Single processor: ( cd src/ngen-forcing && FORCING_PYTEST_WRITE_TEST_EXPECTED_DATA=true pytest tests/input_forcing --map_old_to_new_var_names False)
Multiple processors: ( cd src/ngen-forcing && FORCING_PYTEST_WRITE_TEST_EXPECTED_DATA=true mpirun -n 2 pytest tests/input_forcing --map_old_to_new_var_names False)
```
## Test Configuration

The test suite is configured via `pytest.ini` at the repository root:

- **Python path**: Set to repository root (`.`)
- **Logging**: Enabled with INFO level (DEBUG available by uncommenting)
- **Verbosity**: Full trace with verbose output (`-vv`)
- **Test paths**: Pre-configured to discover tests in `esmf_regrid`, `geomod`, `input_forcing`, and `bmi_model`


## Test Data

Test data is stored in the `test_data/` directory. Tests may reference files from this location for input data and expected results validation.

## Writing New Tests

When adding new tests:

1. Place test files in the appropriate subdirectory
2. Name test files with the `test_*.py` prefix
3. Name test functions with the `test_*` prefix
4. Use fixtures from `conftest.py` for common setup
5. Place test data files in `test_data/` with descriptive names

Example test structure:

```python
import pytest

def test_my_feature():
    """Test description."""
    # Arrange
    input_data = ...

    # Act
    result = function_under_test(input_data)

    # Assert
    assert result == expected_output
```
