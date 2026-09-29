"""Constants shared by coastal forcing utilities."""

from pathlib import Path

ANALYSIS_FLAG = "AnAFlag"
FORECAST_INPUT_HORIZONS = "ForecastInputHorizons"
REFERENCE_FORECAST_DATE = "RefcstBDateProc"
LOOKBACK = "LookBack"
GRID_TYPE = "GRID_TYPE"
GRIDDED_GRID_TYPE = "gridded"
OUTPUT = "Output"
GEOPACKAGE = "Geopackage"
GEOGRID_INPUT = "GeogridIn"
SPATIAL_METADATA_INPUT = "SpatialMetaIn"
LONGITUDE_VARIABLE = "LONVAR"
LATITUDE_VARIABLE = "LATVAR"
REUSE_REGRID_WEIGHTS = "ReuseRegridWeights"
REGRID_WEIGHTS_DIR = "RegridWeightsDir"
SCRATCH_DIRECTORY = "ScratchDir"
INPUT_FORCINGS = "InputForcings"
SUPPLEMENTAL_PRECIPITATION = "SuppPcp"

GRID_LONGITUDE_VARIABLE = "XLONG_M"
GRID_LATITUDE_VARIABLE = "XLAT_M"
DEFAULT_FORCING_ROOT_DIR = Path("/ngen-app/data")
DEFAULT_FORCING_TEMPLATE_DIR = (
    Path(__file__).resolve().parents[1] / "BMI_NextGen_Configs" / "config_templates"
)
DEFAULT_GEOGRID_DIR = DEFAULT_FORCING_ROOT_DIR / "esmf_mesh" / "NWM" / "domain"
DEFAULT_REGRID_WEIGHTS_DIR = DEFAULT_FORCING_ROOT_DIR / "esmf_mesh" / "regrid_weights"
DEBUG_CONUS_SUBSET = "CONUS_debug_gauge_01123000"
SPATIAL_METADATA_SUFFIX = {
    "CONUS": "CONUS",
    "Alaska": "AK",
    "Hawaii": "HI",
    "Puerto_Rico": "PRVI",
}

INPUT_PRODUCT_FIELDS = (
    INPUT_FORCINGS,
    "InputForcingDirectories",
    "InputForcingTypes",
    "InputMandatory",
    "ForecastInputHorizons",
    "ForecastInputOffsets",
    "IgnoredBorderWidths",
    "RegridOpt",
    "ForcingTemporalInterpolation",
    "TemperatureBiasCorrection",
    "PressureBiasCorrection",
    "HumidityBiasCorrection",
    "WindBiasCorrection",
    "SwBiasCorrection",
    "LwBiasCorrection",
    "PrecipBiasCorrection",
    "TemperatureDownscaling",
    "ShortwaveDownscaling",
    "PressureDownscaling",
    "PrecipDownscaling",
    "HumidityDownscaling",
    "DownscalingParamDirs",
    "custom_input_fcst_freq",
)

SUPPLEMENTAL_PRODUCT_FIELDS = (
    SUPPLEMENTAL_PRECIPITATION,
    "SuppPcpForcingTypes",
    "SuppPcpDirectories",
    "RegridOptSuppPcp",
    "SuppPcpTemporalInterpolation",
    "SuppPcpInputOffsets",
    "SuppPcpMandatory",
)
