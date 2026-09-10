import importlib.util
import os

import pytest

### Load import tests.test_utils as test_utils, referring explicitly to its path.
### This explicit load is necessary since March 2026 versions of ngen which introduced /ngen-app/ngen/extern/topoflow-glacier/tests
spec = importlib.util.spec_from_file_location(
    "tests.test_utils", os.path.abspath("tests/test_utils.py")
)
test_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test_utils)

consts = test_utils.test_consts
configs = test_utils.test_config_classes
ClassAttrFetcher = test_utils.ClassAttrFetcher

TEST_FILE_NAME_PREFIX = "input_forcing"


TEST_CONFIGS = [
    configs.TestConfig_InputForcing(
        config_file=consts.RETRO_FORCING_CONFIG_FILE__AORC_CONUS,
        keys_to_check=consts.COMPOSITE_KEYS_TO_CHECK,
        keys_to_exclude=tuple(
            set(consts.KEYS_TO_EXCLUDE) | {"config_options", "geo_meta", "mpi_config"}
        ),
        grid_type=consts.GRID_TYPE,
        force_key=12,
        test_file_name_prefix=TEST_FILE_NAME_PREFIX,
        extra_attrs=[ClassAttrFetcher("bmi_model_values", "CAT-ID")],
    ),
]


@pytest.mark.parametrize(
    "bmi_forcing_fixture_input_forcing", TEST_CONFIGS, indirect=True
)
def test_input_forcing(
    bmi_forcing_fixture_input_forcing: test_utils.BMIForcingFixture_InputForcing,  # pyright: ignore
) -> None:
    """Pytest function for testing InputForcing functionality."""
    ### Total number of timesteps needs to be at least 3, since the 1st and 2nd behaves differently than the others,
    ### e.g. see `if config_options.current_output_step == 1` throughout the code and the regridded_forcings1 vs regridded_forcings2 weighting.
    total_timesteps = 3

    fixt = bmi_forcing_fixture_input_forcing
    if len(fixt.input_forcing_mod) != 1:
        raise ValueError(
            f"Expected 1 key for input_forcing_mod, got {len(fixt.input_forcing_mod)}: {list(fixt.input_forcing_mod.keys())}"
        )

    fixt.after_intitialization_check()
    for i in range(total_timesteps):
        fixt.bmi_model.update()
        fixt.after_bmi_model_update(
            current_output_step=i + 1,
        )
    fixt.bmi_model.finalize()
    fixt.after_finalize()
