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

    def effective_lagback(self) -> int:
        """Determines effective lag back window to avoid unpopulated upstream cycles."""
        return getattr(self, "_override_lagback", self.lagback_hours)

    @classmethod
    def from_cli_args(cls) -> typing.Self:
        """Create an instance of the subclass using command-line arguments."""
        parser = argparse.ArgumentParser()
        parser.add_argument("outDir", type=str, help="Output directory path")
        parser.add_argument("startTime", type=lambda s: datetime.strptime(s, "%Y-%m-%d %H:%M:%S"))
        parser.add_argument("--lookBackHours", type=int, default=cls.default_lookback)
        parser.add_argument("--cleanBackHours", type=int, default=cls.default_cleanback)
        parser.add_argument("--lagBackHours", type=int, default=cls.default_lagback)
        parser.add_argument("--ensNumber", type=int, default=None)
        parser.add_argument("--inputforcing", type=int, required=True)
        parser.add_argument("--max_download_attempts", type=int, default=10, help="Number of attempts to download a file.")
        parser.add_argument("--download_attempt_interval", type=int, default=30, help="Wait time (s) between retries.")
        parser.add_argument("--check_file_availability", type=int, default=0, help="Check upstream availability only.")
        parser.add_argument("--subhourly_avg", action="store_true", default=False, help="Flag sub-hourly averaging.")
        parser.add_argument("--gfs_extended", action="store_true", default=False, help="Flag GFS extended horizon.")
        args = parser.parse_args()
        print(f"{cls.__name__} args:", vars(args))

        return cls(
            out_dir=args.outDir,
            start_time=args.startTime,
            lookback_hours=args.lookBackHours,
            cleanback_hours=args.cleanBackHours,
            lagback_hours=args.lagBackHours,
            ens_number=args.ensNumber,
            input_forcing=args.inputforcing,
            max_download_attempts=args.max_download_attempts,
            download_attempt_interval=args.download_attempt_interval,
            check_file_availability=args.check_file_availability,
            subhourly_avg=args.subhourly_avg,
            gfs_extended=args.gfs_extended,
        )

    def run(self) -> None:
        """Execute cleanup and downloads or verify file availability."""
        if self.check_file_availability == 1:
            self._check_data_availability()
        else:
            self._cleanup_old_data()
            self._download_data()

    @property
    @abstractmethod
    def base_url(self) -> str:
        """Return base URL for downloading forecast files."""
        pass

    @abstractmethod
    def get_download_targets(self, d_start: datetime) -> list[typing.Any]:
        """Return list of download targets for a given forecast cycle time."""
        pass

    def should_process_hour(self, d_start: datetime) -> bool:
        """Determine whether a given forecast cycle hour should be processed."""
        return True

    @abstractmethod
    def build_output_dir(self, d_start: datetime, ens_number: str) -> str:
        """Return output directory for the given forecast cycle datetime."""
        pass

    @abstractmethod
    def build_file_url_and_name(
        self, d_start: datetime, target: typing.Any, ens_number: str
    ) -> tuple[str, str] | list[tuple[str, str]]:
        """Construct remote file URL and local target filename."""
        pass

    @property
    def recursive_cleanup(self) -> bool:
        """If True, recursively delete directories and prune parent paths."""
        return False

    def pre_download_hook(self, d_start: datetime) -> None:
        """Hook called before downloading begins for a specific cycle."""
        pass

    def post_download_hook(self, d_start: datetime) -> None:
        """Hook called after downloading completes for a specific cycle."""
        pass

    def _cleanup_old_data(self) -> None:
        """Clean up old forecast cycles using timestamp or recursive pruning."""
        for hour in range(self.cleanback_hours, self.lookback_hours, -1):
            d_start: datetime = self.start_time - timedelta(hours=hour)
            if not self.should_process_hour(d_start):
                continue

            if self.recursive_cleanup:
                leaf_dir: str = self.build_output_dir(d_start, self.ens_number)
                self._remove_dir_and_empty_parents(leaf_dir, levels=2)
            else:
                dir_path: str = self.build_output_dir(d_start, self.ens_number)
                if os.path.isdir(dir_path):
                    LOG.debug(f"Removing old data: {dir_path}")
                    shutil.rmtree(dir_path)
