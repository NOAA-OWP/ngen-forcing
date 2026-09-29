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

TEST_FILE_NAME_PREFIX = "supp_precip"

### This disables a LOG call which was causing a crash at ioMod.py: LOG.debug(f"Wgrib2 command: {Wgrib2Cmd}", True)
os.environ["MFE_SILENT"] = "true"

TEST_CONFIGS = [
    configs.TestConfig_SuppPrecip(
        config_file=consts.FORECAST_FORCING_CONFIG_FILE__SHORT_RANGE_PR,
        keys_to_check=consts.COMPOSITE_KEYS_TO_CHECK,
        # supplemental_precip uses old attribute names, so exclude those old names here
        keys_to_exclude=tuple(
            set(consts.KEYS_TO_EXCLUDE)
            | {
                "config_options",
                "geo_meta",
                "mpi_config",
                "userCycleOffset",
                "timeInterpOpt",
                "inDir",
                "file_type",
                "enforce",
                "regridOpt",
                # non-deterministic
                "esmf_field_out._data",
            }
        ),
        grid_type=consts.GRID_TYPE,
        force_key=15,
        test_file_name_prefix=TEST_FILE_NAME_PREFIX,
        map_old_to_new_var_names=False,
        keys_no_hash=("final_supp_precip",),
        keys_to_exclude_at_init=("regridded_mask",),
        extra_attrs=[ClassAttrFetcher("bmi_model_values", "CAT-ID")],
    ),
]


@pytest.mark.parametrize("bmi_forcing_fixture_supp_precip", TEST_CONFIGS, indirect=True)
def test_supp_precip(
    bmi_forcing_fixture_supp_precip: test_utils.BMIForcingFixture_SuppPrecip,  # pyright: ignore
) -> None:
    """Pytest function for testing SuppPrecip functionality."""
    ### Total number of timesteps needs to be at least 2, since the 1st one behaves differently than the others, e.g. see `if config_options.current_output_step == 1` throughout the code.
    total_timesteps = 3

    fixt = bmi_forcing_fixture_supp_precip
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
