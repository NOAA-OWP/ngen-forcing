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


### This disables a LOG call which was causing a crash at ioMod.py: LOG.debug(f"Wgrib2 command: {Wgrib2Cmd}", True)
os.environ["MFE_SILENT"] = "true"


RETRO_FORCING_CONFIG_FILE__AORC_CONUS = (
    "/workspaces/nwm-rte/src/ngen-forcing/tests/test_data/configs/aorc_config.yml"
)
FORECAST_FORCING_CONFIG_FILE__SHORT_RANGE_CONUS = "/workspaces/nwm-rte/src/ngen-forcing/tests/test_data/configs/short_range_config.yml"
COMPOSITE_KEYS_TO_CHECK = ()
GRID_TYPE = "hydrofabric"  # ["gridded","hydrofabric","unstructured"]
KEYS_TO_EXCLUDE = ("uid64",)


@pytest.mark.parametrize(
    "bmi_forcing_fixture_geomod",
    [
        (
            RETRO_FORCING_CONFIG_FILE__AORC_CONUS,
            COMPOSITE_KEYS_TO_CHECK,
            KEYS_TO_EXCLUDE,
            GRID_TYPE,
        )
    ],
    indirect=True,
)
def test_geomod(
    bmi_forcing_fixture_geomod: test_utils.BMIForcingFixture_GeoMod,  # pyright: ignore
) -> None:
    """Pytest function for testing GeoMod functionality."""
    ### Total number of timesteps needs to be at least 2, since the 1st one behaves differently than the others, e.g. see `if config_options.current_output_step == 1` throughout the code.
    total_timesteps = 3

    fixt = bmi_forcing_fixture_geomod
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
