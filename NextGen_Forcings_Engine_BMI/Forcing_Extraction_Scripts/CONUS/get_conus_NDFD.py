"""
National Digital Forecast Database (NDFD) CONUS operational forcing downloader.
Extracts and concatenates operational GRIB2 bins for temperature, wind, and precipitation.
"""

from __future__ import annotations

import datetime
import logging
import os
from pathlib import Path
import typing
import requests

from Forcing_Extraction_Scripts.forecast_download_base import ForecastDownloader

logger = logging.getLogger(__name__)


class NDFDConusDownloader(ForecastDownloader):
    """Downloader implementation for operational CONUS NDFD forecasts."""

    FIELD_MAP: dict[str, str] = {
        "TMP": "ds.temp.bin",
        "WDIR": "ds.wdir.bin",
        "WSPD": "ds.wspd.bin",
        "APCP": "ds.qpf.bin",
    }

    VALID_PERIODS: list[str] = ["VP.001-003", "VP.004-007"]

    @property
    def base_url(self) -> str:
        return "https://noaa-ndfd-pds.s3.amazonaws.com/opnl/AR.conus"

    def should_process_hour(self, d_current: datetime.datetime) -> bool:
        """Process 6-hourly operational cycles (00, 06, 12, 18 UTC)."""
        return d_current.hour in (0, 6, 12, 18)

    def get_valid_cycles(self, cycle_time: datetime.datetime) -> list[datetime.datetime]:
        """NDFD operates continuously; cycles align to 6-hour NextGen steps."""
        return [cycle_time]

    def get_download_targets(self, d_current: datetime.datetime) -> list[str]:
        """Return the target variable names expected by NextGen."""
        return list(self.FIELD_MAP.keys())

    def build_output_dir(self, d_current: datetime.datetime, _=None) -> str:
        """Construct <inDir>/NDFD/YYYYMMDD/wgrbbul/ndfd_conus."""
        date_str = d_current.strftime("%Y%m%d")
        return os.path.join(self.out_dir, "NDFD", date_str, "wgrbbul", "ndfd_conus")

    def build_file_url_and_name(
        self,
        d_current: datetime.datetime,
        target: str,
        _=None,
    ) -> tuple[str, str]:
        """Return primary VP.001-003 URL for availability ping and local filename."""
        bin_file = self.FIELD_MAP.get(target, f"ds.{target.lower()}.bin")
        url = f"{self.base_url}/VP.001-003/{bin_file}"
        filename = f"ndfd_conus_{target}.grib2"
        return url, filename
