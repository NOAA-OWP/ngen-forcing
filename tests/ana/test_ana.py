import importlib.util
import logging
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


TEST_CONFIGS = [
    configs.TestConfig_AnA(
        config_file=consts.FORECAST_FORCING_CONFIG_FILE__ANA_CONUS,
        keys_to_check=(),
        keys_to_exclude=consts.KEYS_TO_EXCLUDE,
        grid_type=consts.GRID_TYPE,
        test_file_name_prefix="ana_standard_conus",
        extra_attrs=[
            ClassAttrFetcher("bmi_model_values", "CAT-ID"),
        ],
    ),
]


@pytest.mark.parametrize("bmi_forcing_fixture_ana", TEST_CONFIGS, indirect=True)
def test_ana(
    bmi_forcing_fixture_ana: test_utils.BMIForcingFixture_AnA,  # pyright: ignore
) -> None:
    """Pytest function for testing Analysis and Assimilation."""
    ### Total number of timesteps needs to be at least 3, since the 1st and 2nd behaves differently than the others,
    ### e.g. see `if config_options.current_output_step == 1` throughout the code and the regridded_forcings1 vs regridded_forcings2 weighting.
    total_timesteps = 3

    fixt = bmi_forcing_fixture_ana
    # fixt.after_intitialization_check()  # NOTE this is disabled because with -n 2, there are attrs initialized to arbitrary values.
    for i in range(total_timesteps):
        logging.info("Starting bmi_model.update()...")
        fixt.bmi_model.update()
        fixt.after_bmi_model_update(
            current_output_step=i + 1,
        )
    logging.info("Starting bmi_model.finalize()...")
    fixt.bmi_model.finalize()
    fixt.after_finalize()
