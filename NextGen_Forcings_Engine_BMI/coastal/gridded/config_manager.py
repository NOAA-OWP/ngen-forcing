"""Build effective forcing configurations for gridded coastal domains.

This was added when adding support for coastal workflows.
The original business logic was in the ``nwm-rte`` repo in ``nwm-rte/run_coastal.py`` in the ``coastalforcing` branch.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from NextGen_Forcings_Engine_BMI.coastal import consts
from NextGen_Forcings_Engine_BMI.run_bmi_model import REFERENCE_TIME_FORMAT

LOG = logging.getLogger("FORCING")


def analysis_num_updates(config_path: Path) -> int | None:
    """Return the hourly AnA iteration count implied by input horizons."""
    config = yaml.safe_load(config_path.read_text())
    if config.get(consts.ANALYSIS_FLAG, 0) != 1:
        return None
    horizons = config.get(consts.FORECAST_INPUT_HORIZONS, [])
    if not horizons:
        raise ValueError(
            f"{consts.FORECAST_INPUT_HORIZONS} is required for analysis cycles"
        )
    num_updates = int(max(horizons) / 60)
    if num_updates < 1:
        raise ValueError(
            f"{consts.FORECAST_INPUT_HORIZONS} must cover at least one hour"
        )
    return num_updates


class CoastalConfigManager:
    """Build a gridded forcing config for one coastal domain."""

    def __init__(
        self,
        forcing_configuration: str,
        global_domain: str,
        cycle_datetime: datetime | None = None,
        start_time: datetime | None = None,
        lookback: int | None = None,
        forecast_input_horizons: int | None = None,
        *,
        forcing_root_dir: str | Path = consts.DEFAULT_FORCING_ROOT_DIR,
        forcing_template_dir: str | Path = consts.DEFAULT_FORCING_TEMPLATE_DIR,
        geogrid_dir: str | Path = consts.DEFAULT_GEOGRID_DIR,
        regrid_weights_dir: str | Path = consts.DEFAULT_REGRID_WEIGHTS_DIR,
        spatial_metadata_suffix: Mapping[str, str] = consts.SPATIAL_METADATA_SUFFIX,
        debug_subset_name: str = consts.DEBUG_CONUS_SUBSET,
        debug_conus_subset: bool = False,
    ) -> None:
        if debug_conus_subset and global_domain != "CONUS":
            raise ValueError("debug_conus_subset requires global_domain='CONUS'")
        self.forcing_configuration = forcing_configuration
        self.global_domain = global_domain
        self.cycle_datetime = cycle_datetime
        self.start_time = start_time
        self.lookback = lookback
        self.forecast_input_horizons = forecast_input_horizons
        self.forcing_root_dir = Path(forcing_root_dir)
        self.forcing_template_dir = Path(forcing_template_dir)
        self.geogrid_dir = Path(geogrid_dir)
        self.regrid_weights_dir = Path(regrid_weights_dir)
        self.spatial_metadata_suffix = spatial_metadata_suffix
        self.debug_subset_name = debug_subset_name
        self.debug_conus_subset = debug_conus_subset

    @property
    def target_domain(self) -> str:
        if self.debug_conus_subset:
            return self.debug_subset_name
        return self.global_domain

    @property
    def template_path(self) -> Path:
        return self.forcing_template_dir / f"{self.forcing_configuration}_config.yml"

    @property
    def geogrid(self) -> Path:
        return self.geogrid_dir / f"geo_em_{self.target_domain}.nc"

    @property
    def spatial_metadata(self) -> Path:
        if self.debug_conus_subset:
            spatial_suffix = self.debug_subset_name
        else:
            spatial_suffix = self.spatial_metadata_suffix.get(
                self.global_domain, self.global_domain
            )
        return (
            self.geogrid_dir / f"GEOGRID_LDASOUT_Spatial_Metadata_{spatial_suffix}.nc"
        )

    @property
    def output_dir(self) -> Path:
        subset_suffix = f"_{self.target_domain}" if self.debug_conus_subset else ""
        return (
            self.forcing_root_dir
            / "scratch"
            / f"{self.forcing_configuration}_coastal{subset_suffix}"
        )

    @property
    def file_date(self) -> datetime:
        file_date = self.cycle_datetime or self.start_time
        if file_date is None:
            raise ValueError("Provide cycle_datetime or start_time and end_time")
        return file_date

    @property
    def output_path(self) -> Path:
        timestamp = self.file_date.strftime(REFERENCE_TIME_FORMAT)
        return self.output_dir / f"{self.target_domain}_{timestamp}.nc"

    @property
    def config_path(self) -> Path:
        return self.output_path.with_suffix(".yml")

    def _validate_paths(self) -> None:
        if not self.template_path.is_file():
            raise FileNotFoundError(
                f"No forcing template found for {self.forcing_configuration!r}: "
                f"{self.template_path}"
            )
        for required_path in (self.geogrid, self.spatial_metadata):
            if not required_path.is_file():
                raise FileNotFoundError(
                    f"Required coastal domain file not found: {required_path}"
                )

    def build_config(self) -> dict[str, Any]:
        """Load the installed template and apply coastal gridded overrides."""
        self._validate_paths()
        template = self.template_path.read_text()
        template = template.replace("{root_dir}", str(self.forcing_root_dir))
        template = template.replace("{global_domain}", self.global_domain)
        config = yaml.safe_load(template)
        config[consts.REFERENCE_FORECAST_DATE] = self.file_date.strftime(
            REFERENCE_TIME_FORMAT
        )

        if (
            self.lookback is not None or self.forecast_input_horizons is not None
        ) and config.get(consts.ANALYSIS_FLAG, 0) != 1:
            raise ValueError(
                "lookback and forecast_input_horizons are only supported for "
                f"analysis configurations. {self.forcing_configuration} has "
                f"{consts.ANALYSIS_FLAG}="
                f"{config.get(consts.ANALYSIS_FLAG, 0)}"
            )
        if self.lookback is not None:
            config[consts.LOOKBACK] = self.lookback
        if self.forecast_input_horizons is not None:
            configured_horizons = config.get(consts.FORECAST_INPUT_HORIZONS, [1])
            config[consts.FORECAST_INPUT_HORIZONS] = [
                self.forecast_input_horizons
            ] * len(configured_horizons)

        config.update(
            {
                consts.GRID_TYPE: consts.GRIDDED_GRID_TYPE,
                consts.OUTPUT: 1,
                consts.GEOPACKAGE: "",
                consts.GEOGRID_INPUT: str(self.geogrid),
                consts.SPATIAL_METADATA_INPUT: str(self.spatial_metadata),
                consts.LONGITUDE_VARIABLE: consts.GRID_LONGITUDE_VARIABLE,
                consts.LATITUDE_VARIABLE: consts.GRID_LATITUDE_VARIABLE,
                consts.REGRID_WEIGHTS_DIR: str(self.regrid_weights_dir),
                consts.REUSE_REGRID_WEIGHTS: True,
                consts.SCRATCH_DIRECTORY: str(self.output_dir),
            }
        )
        return config

    def write_config(self) -> Path:
        """Write the effective config beside the final grid and return its path."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.regrid_weights_dir.mkdir(parents=True, exist_ok=True)
        config = self.build_config()
        with self.config_path.open("w") as config_file:
            yaml.safe_dump(config, config_file, sort_keys=False)
        return self.config_path
