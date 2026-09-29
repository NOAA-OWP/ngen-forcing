import os

### This disables a LOG call which was causing a crash at ioMod.py: LOG.debug(f"Wgrib2 command: {Wgrib2Cmd}", True)
os.environ["MFE_SILENT"] = "true"


RETRO_FORCING_CONFIG_FILE__AORC_CONUS = (
    "/workspaces/nwm-rte/src/ngen-forcing/tests/test_data/configs/aorc_config.yml"
)
FORECAST_FORCING_CONFIG_FILE__SHORT_RANGE_CONUS = "/workspaces/nwm-rte/src/ngen-forcing/tests/test_data/configs/short_range_config.yml"
FORECAST_FORCING_CONFIG_FILE__SHORT_RANGE_PR = "/workspaces/nwm-rte/src/ngen-forcing/tests/test_data/configs/short_range_puertorico_config.yml"
FORECAST_FORCING_CONFIG_FILE__ANA_CONUS = "/workspaces/nwm-rte/src/ngen-forcing/tests/test_data/configs/standard_ana_config.yml"

COMPOSITE_KEYS_TO_CHECK = ()
GRID_TYPE = "hydrofabric"  # ["gridded","hydrofabric","unstructured"]
KEYS_TO_EXCLUDE = ("uid64",)
