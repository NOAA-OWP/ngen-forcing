# Tests README

This directory contains tests for the NextGen Forcing BMI Engine.

## Initial test data

Tests data is included in the `test_data` directory and includes configs, gpkgs, esmf_meshes, expected results and actual results. While the configs, gpkgs, esmf_meshes and expectd results are included in the repo and can be used as is, the following steps can be taken to re-create these test inputs.

---
The initial test data was generated using `nwm-rte` to create a calibration realization
for gage 01123000, starting at time 2013-07-01 00:00:00, and running for 3 timesteps,
using `nwm-rte's` run_suite.sh.  See RETRO_FORCING_CONFIG_FILE__AORC_CONUS.


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
- **`supp_precip/`** - Tests for supp precip data processing
- **`bmi_model/`** - Tests for the BMI model lifecycle
- **`config_options/`** - Tests config options
- **`test_utils.py`** - Shared test utilities and fixtures
- **`conftest.py`** - Pytest configuration and shared fixtures

## Prerequisite Steps
    1. Clone the nwm-rte repository
    2. Build a Docker image using nwm-rte.
    3. Enter a Dev Container using nwm-rte.

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

# Supp precip tests
Single processor: ( cd src/ngen-forcing && pytest tests/supp_precip)
Multiple processors: ( cd src/ngen-forcing && mpirun -n 2 pytest tests/supp_precip)

# Analysis and Assimilation tests
Single processor: ( cd src/ngen-forcing && pytest tests/ana )
Multiple processors: ( cd src/ngen-forcing && mpirun -n 2 pytest tests/ana )

# BMI model tests
Single processor: ( cd src/ngen-forcing && pytest tests/bmi_model)
Multiple processors: ( cd src/ngen-forcing && mpirun -n 2 pytest tests/bmi_model)

# config options tests
Single processor: ( cd src/ngen-forcing && pytest tests/config_options)
Multiple processors: ( cd src/ngen-forcing && mpirun -n 2 pytest tests/config_options)
```

Create new test output data (creates expected outputs for subsequent tests)
```bash
# All
Single processor: (cd src/ngen-forcing && FORCING_PYTEST_WRITE_TEST_EXPECTED_DATA=true pytest )
Multiple processors: ( cd src/ngen-forcing && FORCING_PYTEST_WRITE_TEST_EXPECTED_DATA=true mpirun -n 2 pytest )

# ESMF regridding tests
Single processor: ( cd src/ngen-forcing && FORCING_PYTEST_WRITE_TEST_EXPECTED_DATA=true pytest tests/esmf_regrid)
Multiple processors: ( cd src/ngen-forcing && FORCING_PYTEST_WRITE_TEST_EXPECTED_DATA=true mpirun -n 2 pytest tests/esmf_regrid)

# GeoMod tests
Single processor: ( cd src/ngen-forcing && FORCING_PYTEST_WRITE_TEST_EXPECTED_DATA=true pytest tests/geomod)
Multiple processors: ( cd src/ngen-forcing && FORCING_PYTEST_WRITE_TEST_EXPECTED_DATA=true mpirun -n 2 pytest tests/geomod)

# Input forcing tests
Single processor: ( cd src/ngen-forcing && FORCING_PYTEST_WRITE_TEST_EXPECTED_DATA=true pytest tests/input_forcing)
Multiple processors: ( cd src/ngen-forcing && FORCING_PYTEST_WRITE_TEST_EXPECTED_DATA=true mpirun -n 2 pytest tests/input_forcing)

# Supp precip tests
Single processor: ( cd src/ngen-forcing && FORCING_PYTEST_WRITE_TEST_EXPECTED_DATA=true pytest tests/supp_precip)
Multiple processors: ( cd src/ngen-forcing && FORCING_PYTEST_WRITE_TEST_EXPECTED_DATA=true mpirun -n 2 pytest tests/supp_precip)

# Analysis and Assimilation tests
Single processor: ( cd src/ngen-forcing && FORCING_PYTEST_WRITE_TEST_EXPECTED_DATA=true pytest tests/ana )
Multiple processors: ( cd src/ngen-forcing && FORCING_PYTEST_WRITE_TEST_EXPECTED_DATA=true mpirun -n 2 pytest tests/ana )

# BMI model tests
Single processor: ( cd src/ngen-forcing && FORCING_PYTEST_WRITE_TEST_EXPECTED_DATA=true pytest tests/bmi_model)
Multiple processors: ( cd src/ngen-forcing && FORCING_PYTEST_WRITE_TEST_EXPECTED_DATA=true mpirun -n 2 pytest tests/bmi_model)

# config_options tests
Single processor: ( cd src/ngen-forcing && FORCING_PYTEST_WRITE_TEST_EXPECTED_DATA=true pytest tests/config_options)
Multiple processors: ( cd src/ngen-forcing && FORCING_PYTEST_WRITE_TEST_EXPECTED_DATA=true mpirun -n 2 pytest tests/config_options)
```

In the rare case where you want to create new `expected` data and run the tests using `old` variable names use the following for `Input Forcing Tests`:
```bash
# Input forcing tests
Single processor: ( cd src/ngen-forcing && FORCING_PYTEST_WRITE_TEST_EXPECTED_DATA=true pytest tests/input_forcing --map_old_to_new_var_names False)
Multiple processors: ( cd src/ngen-forcing && FORCING_PYTEST_WRITE_TEST_EXPECTED_DATA=true mpirun -n 2 pytest tests/input_forcing --map_old_to_new_var_names False)

# Supp precip tests
Single processor: ( cd src/ngen-forcing && FORCING_PYTEST_WRITE_TEST_EXPECTED_DATA=true pytest tests/supp_precip --map_old_to_new_var_names False)
Multiple processors: ( cd src/ngen-forcing && FORCING_PYTEST_WRITE_TEST_EXPECTED_DATA=true mpirun -n 2 pytest tests/supp_precip --map_old_to_new_var_names False)

```
## Test Configuration

The test suite is configured via `pytest.ini` at the repository root.

## Test Data

Test data is stored in the `test_data/` directory. Tests may reference files from this location for input data and expected results validation.

## Writing New Tests or Updating Expected Results Files

When adding new tests, use the OS env var `FORCING_PYTEST_WRITE_TEST_EXPECTED_DATA`
to have your new test automatically write new "expected" data to `tests/test_data/expected_results/`,
then commit those files to the repository.  See above for example calls.

## Key Exclusion in Test Configs

`keys_to_exclude` and `keys_to_exclude_at_init` both support dot notation to target nested keys.
A plain key like `"config_options"` is removed at every nesting level.
A dotted key like `"esmf_field_out._data"` removes only `_data` when it appears
directly under `esmf_field_out`, leaving all other occurrences of `_data` intact.

This means that the exclusion logic does not support the notion of a key that contains a literal dot in its name,
since a dot always causes a hierarchical search.
