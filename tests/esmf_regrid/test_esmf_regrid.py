import importlib.util
import logging
import os

import pytest

from NextGen_Forcings_Engine_BMI.NextGen_Forcings_Engine.core.regrid import (
    regrid_aorc_aws,
    regrid_conus_hrrr,
    regrid_conus_rap,
)

### Load import tests.test_utils as test_utils, referring explicitly to its path.
### This explicit load is necessary since March 2026 versions of ngen which introduced /ngen-app/ngen/extern/topoflow-glacier/tests
spec = importlib.util.spec_from_file_location(
    "tests.test_utils", os.path.abspath("tests/test_utils.py")
)
test_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test_utils)

ClassAttrFetcher = test_utils.ClassAttrFetcher

### This disables a LOG call which was causing a crash at ioMod.py: LOG.debug(f"Wgrib2 command: {Wgrib2Cmd}", True)
os.environ["MFE_SILENT"] = "true"


RETRO_FORCING_CONFIG_FILE__AORC_CONUS = (
    "/workspaces/nwm-rte/src/ngen-forcing/tests/test_data/configs/aorc_config.yml"
)
FORECAST_FORCING_CONFIG_FILE__SHORT_RANGE_CONUS = "/workspaces/nwm-rte/src/ngen-forcing/tests/test_data/configs/short_range_config.yml"


### These are output arrays which can contain extra unused elements which need to be removed during an equality check.
REGRID_ARRAYS_TO_TRIM_EXTRA_ELEMENTS: tuple[str] = (
    "regridded_forcings1",
    "regridded_forcings1_elem",
    "regridded_forcings2",
    "regridded_forcings2_elem",
)

### These are keys to include in the "expected" test results json, and are checked for equality versus "actual" results from regrid operation.
### These are gathered from the resulting InputForcings class instance.
REGRID_KEYS_TO_CHECK: tuple[str] = REGRID_ARRAYS_TO_TRIM_EXTRA_ELEMENTS + (
    # "esmf_field_in",
    # "esmf_field_in_elem",
    # "esmf_grid_in",
    # "esmf_grid_in_elem",
    # "esmf_field_out",
    # "esmf_field_out_elem",
    "regridded_mask",
    ### TODO revisit to see which use cases require checking this. Some notes in the code indicate that it is not used, but this has not been confirmed globally.
    # "regridded_mask_AORC",
    "regridded_mask_elem",
    "regridded_mask_elem_AORC",
    "regridded_precip1",
    "regridded_precip1_elem",
    "regridded_precip2",
    "regridded_precip2_elem",
)

### While the InputForcings class instance is the primary source of test results data,
### this is used to add supplemental attributes to the results data,
### for example "element_ids" (for hydrofabric discretization, these are catchment IDs).
EXTRA_ATTRS: tuple[ClassAttrFetcher] = (ClassAttrFetcher("geo_meta", "element_ids"),)

COMPOSITE_KEYS_TO_CHECK: tuple[str] = REGRID_KEYS_TO_CHECK + tuple(
    _.results_key_name for _ in EXTRA_ATTRS
)
GRID_TYPE = "hydrofabric"  # ["gridded","hydrofabric","unstructured"]
KEYS_TO_EXCLUDE = ("uid64",)


@pytest.mark.parametrize(
    "bmi_forcing_fixture_regrid",
    [
        (
            regrid_aorc_aws,
            RETRO_FORCING_CONFIG_FILE__AORC_CONUS,
            12,
            EXTRA_ATTRS,
            REGRID_ARRAYS_TO_TRIM_EXTRA_ELEMENTS,
            COMPOSITE_KEYS_TO_CHECK,
            KEYS_TO_EXCLUDE,
            GRID_TYPE,
        ),
        (
            regrid_conus_hrrr,
            FORECAST_FORCING_CONFIG_FILE__SHORT_RANGE_CONUS,
            5,
            EXTRA_ATTRS,
            REGRID_ARRAYS_TO_TRIM_EXTRA_ELEMENTS,
            COMPOSITE_KEYS_TO_CHECK,
            KEYS_TO_EXCLUDE,
            GRID_TYPE,
        ),
        (
            regrid_conus_rap,
            FORECAST_FORCING_CONFIG_FILE__SHORT_RANGE_CONUS,
            6,
            EXTRA_ATTRS,
            REGRID_ARRAYS_TO_TRIM_EXTRA_ELEMENTS,
            COMPOSITE_KEYS_TO_CHECK,
            KEYS_TO_EXCLUDE,
            GRID_TYPE,
        ),
    ],
    indirect=True,
)
def test_regrid(
    bmi_forcing_fixture_regrid: test_utils.BMIForcingFixture_Regrid,  # pyright: ignore
) -> None:
    """pytest function for testing ESMF regrid functionality.
    NOTE vvv this has been tested for the following conditions only vvv
        1. Hydrofabric discretization, AORC historical forcing, CONUS domain.
        2. Hydrofabric discretization, HRRR and RAP forcing (individually), CONUS domain.
    NOTE ^^^ this has been tested for the above conditions only ^^^
    """
    ### Total number of timesteps needs to be at least 2, since the 1st one behaves differently than the others, e.g. see `if config_options.current_output_step == 1` throughout the code.
    total_timesteps = 3

    fixt = bmi_forcing_fixture_regrid
    if len(fixt.input_forcing_mod) != 1:
        raise ValueError(
            f"Expected 1 key for input_forcing_mod, got {len(fixt.input_forcing_mod)}: {list(fixt.input_forcing_mod.keys())}"
        )

    input_forcings = fixt.input_forcing_mod[fixt.force_key]

    for i in range(total_timesteps):
        fixt.pre_regrid()
        fixt.run_regrid(input_forcings)
        fixt.check_regrid_results(input_forcings)
        fixt.post_regrid()
