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
