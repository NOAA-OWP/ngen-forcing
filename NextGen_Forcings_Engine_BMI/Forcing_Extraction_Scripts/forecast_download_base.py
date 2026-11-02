from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import time
import typing
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from urllib import error, request

import requests

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

LOG = logging.getLogger()


class ForecastDownloader(ABC):
    """Abstract base class for structured forecast data downloads."""

    default_lookback: int = 24
    default_cleanback: int = 240
    default_lagback: int = 6

    def __init__(
        self,
        out_dir: str | os.PathLike,
        start_time: datetime,
        lookback_hours: int,
        cleanback_hours: int,
        lagback_hours: int,
        ens_number: int | str | None,
        input_forcing: int,
        max_download_attempts: int,
        download_attempt_interval: int,
        check_file_availability: int,
        subhourly_avg: bool,
        gfs_extended: bool,
    ) -> None:
        if lookback_hours <= lagback_hours:
            raise ValueError(
                f"Invalid configuration: lookback_hours ({lookback_hours}) must be greater than "
                f"lagback_hours ({lagback_hours}) to allow for a valid processing range."
            )
        self.out_dir: str | os.PathLike = out_dir
        self.start_time: datetime = start_time
        self.lookback_hours: int = lookback_hours
        self.cleanback_hours: int = cleanback_hours
        self.lagback_hours: int = lagback_hours
        self.ens_number: str = str(ens_number).zfill(2) if ens_number is not None else "00"
        self.input_forcing: int = input_forcing
        self.max_download_attempts: int = max_download_attempts
        self.download_attempt_interval: int = download_attempt_interval
        self.check_file_availability: int = check_file_availability
        self.subhourly_avg: bool = subhourly_avg
        self.gfs_extended: bool = gfs_extended

        self.last_hour: bool | None = None
        self.d_now: datetime = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

        os.makedirs(self.out_dir, exist_ok=True)
