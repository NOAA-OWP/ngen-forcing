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

    def iter_remote_files(self) -> typing.Iterator[tuple[datetime, str, str]]:
        """Yield (cycle time, remote URL, local path) for every file the processing window needs."""
        for hour in range(self.lookback_hours, self.effective_lagback(), -1):
            d_start = self.start_time - timedelta(hours=hour)
            if not self.should_process_hour(d_start):
                continue
            output_dir = self.build_output_dir(d_start, self.ens_number)
            for target in self.get_download_targets(d_start):
                result = self.build_file_url_and_name(d_start, target, self.ens_number)
                for url, filename in (result if isinstance(result, list) else [result]):
                    yield d_start, url, os.path.join(output_dir, filename)

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

    @staticmethod
    def _remove_dir_and_empty_parents(path: str, levels: int = 2) -> None:
        """Removes a directory and prunes up to levels empty parent directories."""
        if os.path.isdir(path):
            LOG.debug(f"Removing directory: {path}")
            shutil.rmtree(path)

            current_path: str = path
            for _ in range(levels):
                current_path = os.path.dirname(current_path)
                if os.path.isdir(current_path) and not os.listdir(current_path):
                    LOG.debug(f"Removing empty parent directory: {current_path}")
                    os.rmdir(current_path)
                else:
                    break

    def _download_data(self) -> None:
        """Download forecast files across desired time horizons and targets."""
        LOG.info(
            f"ForecastDownloader: Download data. lookback: {self.lookback_hours} "
            f"lagback: {self.effective_lagback()}"
        )
        final_hour: int = self.effective_lagback() + 1

        for hour in range(self.lookback_hours, self.effective_lagback(), -1):
            d_start: datetime = self.start_time - timedelta(hours=hour)

            if self.should_process_hour(d_start):
                LOG.debug(f"Processing hour offset: {hour}, timestamp: {d_start}")
            else:
                LOG.debug(f"Skipping hour offset: {hour}, timestamp: {d_start}")
                continue

            output_dir: str = self.build_output_dir(d_start, self.ens_number)
            os.makedirs(output_dir, exist_ok=True)

            self.pre_download_hook(d_start)
            targets: list[typing.Any] = self.get_download_targets(d_start)
            self.last_hour = (hour == final_hour)

            # Special handling for RAP (input_forcing == 6): RAP requires downloading
            # multiple file extensions (["pgrb", "bgrb"]) to assemble all mandatory
            # atmospheric and surface forcing fields into NextGen Forcings Engine BMI.
            if self.input_forcing == 6:
                for target in targets:
                    file_list: list[tuple[str, str]] = typing.cast(
                        list[tuple[str, str]], self.build_file_url_and_name(d_start, target, self.ens_number)
                    )
                    for url, filename in file_list:
                        out_path: str = os.path.join(output_dir, filename)
                        LOG.info(f"Looking for file {out_path}")
                        if os.path.isfile(out_path):
                            LOG.info(f"Skipping existing: {out_path}")
                            continue
                        self._download_file(url, out_path)
                self.post_download_hook(d_start)
            else:
                for target in targets:
                    url, filename = typing.cast(
                        tuple[str, str], self.build_file_url_and_name(d_start, target, self.ens_number)
                    )
                    out_path = os.path.join(output_dir, filename)
                    LOG.info(f"Looking for file {out_path}")
                    if os.path.isfile(out_path):
                        LOG.info(f"Skipping existing: {out_path}")
                        continue
                    self._download_file(url, out_path)
                self.post_download_hook(d_start)

    def _download_file(self, url: str, out_path: str) -> None:
        """Download remote URL to temporary path and publish atomically."""
        max_attempts: int = self.max_download_attempts
        interval: int = self.download_attempt_interval
        attempt: int = 0

        out_dir: str = os.path.dirname(out_path)
        base: str = os.path.basename(out_path)
        temp_path: str = os.path.join(out_dir, f".{base}.tmp.{uuid.uuid4()}")

        while attempt < max_attempts:
            try:
                LOG.info(f"Attempt {attempt + 1}: Downloading {url} -> {temp_path}")
                request.urlretrieve(url, temp_path)
                try:
                    os.link(temp_path, out_path)
                    os.remove(temp_path)
                    LOG.info(f"Download complete: {out_path}")
                    return
                except FileExistsError:
                    LOG.info(f"{out_path} already exists; another process wrote it first. Removing temp.")
                    os.remove(temp_path)
                    return
            except error.HTTPError as e:
                LOG.error(f"HTTPError {e.code} while downloading {url}: {e.reason}")
            except error.URLError as e:
                LOG.error(f"URLError while downloading {url}: {e.reason}")
            except Exception as e:
                LOG.error(f"Unexpected error while downloading {url}: {e}")
            finally:
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass
            attempt += 1
            time.sleep(interval)

        if self.last_hour and self.subhourly_avg:
            LOG.warning(f"Failed to download after {max_attempts} attempts: {url}")
            LOG.warning(
                "This MRMS file is not required to fill in the latest AnA cycle hour for "
                "sub-hourly averaging method. Will use HRRR instead."
            )
        else:
            LOG.error(f"Failed to download after {max_attempts} attempts: {url}")
            LOG.critical(f"Script terminating with exit code 1 due to 404 error on {url}")
            os._exit(1)

    def _check_data_availability(self) -> None:
        """Check forecast file availability without fetching content."""
        LOG.info(
            f"ForecastDownloader: Checking data availability. lookback: {self.lookback_hours} "
            f"lagback: {self.effective_lagback()}"
        )
        total_files_checked: int = 0
        available_files_count: int = 0
        final_hour: int = self.effective_lagback() + 1

        for hour in range(self.lookback_hours, self.effective_lagback(), -1):
            d_start: datetime = self.start_time - timedelta(hours=hour)
            if not self.should_process_hour(d_start):
                continue

            self.pre_download_hook(d_start)
            targets: list[typing.Any] = self.get_download_targets(d_start)
            self.last_hour = (hour == final_hour)
            all_files_available: bool = True

            # Special handling for RAP (input_forcing == 6): verify upstream availability
            # across all required file extensions (["pgrb", "bgrb"]) returned as (url, filename) pairs.
            if self.input_forcing == 6:
                for target in targets:
                    file_list = typing.cast(
                        list[tuple[str, str]], self.build_file_url_and_name(d_start, target, self.ens_number)
                    )
                    for url, filename in file_list:
                        total_files_checked += 1
                        LOG.info(f"Checking upstream availability for {filename} via HEAD request")
                        if self._check_file_availability(url):
                            available_files_count += 1
                        else:
                            all_files_available = False
                            LOG.warning(f"Required file missing from server: {url}")
            else:
                for target in targets:
                    url, filename = typing.cast(
                        tuple[str, str], self.build_file_url_and_name(d_start, target, self.ens_number)
                    )
                    total_files_checked += 1
                    LOG.info(f"Checking upstream availability for {filename} via HEAD request")
                    if self._check_file_availability(url):
                        available_files_count += 1
                    else:
                        all_files_available = False
                        LOG.warning(f"Required file missing from server: {url}")

            if all_files_available:
                LOG.info(f"All required forcing files are available for timestamp: {d_start}")
                self.post_download_hook(d_start)
            else:
                LOG.error(f"One or more forcing files missing for timestamp: {d_start}.")

        LOG.info("--- Forcing Availability Verification Summary ---")
        LOG.info(f"Total files checked: {total_files_checked}")
        LOG.info(f"Total files available on server: {available_files_count}")

        if available_files_count == total_files_checked:
            LOG.info("Verification complete: 100% of requested files are available upstream.")
        else:
            missing_count: int = total_files_checked - available_files_count
            LOG.warning(f"Verification complete: {missing_count} file(s) are currently missing from the server.")

    def _check_file_availability(self, url: str) -> bool:
        """Ping server using HTTP HEAD request to verify file availability."""
        max_attempts: int = self.max_download_attempts
        interval: int = self.download_attempt_interval
        attempt: int = 0

        while attempt < max_attempts:
            try:
                LOG.info(f"Attempt {attempt + 1}: Pinging {url} for availability...")
                response: requests.Response = requests.head(url, timeout=10)

                if response.status_code == requests.codes.ok:
                    LOG.info(f"File is available: {url}")
                    return True
                elif response.status_code == 404:
                    LOG.warning(f"File not found (404): {url}")
                else:
                    LOG.error(f"Server returned status code {response.status_code} while pinging {url}")

            except requests.exceptions.Timeout:
                LOG.error(f"Timeout occurred while trying to connect to {url}")
            except requests.exceptions.ConnectionError:
                LOG.error(f"ConnectionError (Network issue) while pinging {url}")
            except requests.exceptions.RequestException as e:
                LOG.error(f"An error occurred with the request: {e}")

            attempt += 1
            if attempt < max_attempts:
                time.sleep(interval)

        if self.last_hour and self.subhourly_avg:
            LOG.warning(f"Failed to confirm availability after {max_attempts} attempts: {url}")
            return False
        else:
            LOG.error(f"Failed to confirm file availability after {max_attempts} attempts: {url}")
            LOG.critical(f"Script terminating with exit code 1 due to 404 error on {url}")
            os._exit(1)


class FixedFileDownloader(ForecastDownloader, ABC):
    """Subclass for MRMS and StageIV datasets with predefined filenames and subdirectories."""

    def build_file_url_and_name(self, d_start: datetime, target: typing.Any, ens_number: str) -> tuple[str, str]:
        raise NotImplementedError("FixedFileDownloader uses get_file_specs() instead.")

    def get_download_targets(self, d_start: datetime) -> list[typing.Any]:
        return []

    @abstractmethod
    def get_file_specs(self, d_start: datetime) -> list[tuple[str, str]]:
        """Return a list of (subdir, filename) tuples for files to be downloaded."""
        pass

    def iter_remote_files(self) -> typing.Iterator[tuple[datetime, str, str]]:
        for hour in range(self.lookback_hours, self.effective_lagback(), -1):
            d_start = self.start_time - timedelta(hours=hour)
            if not self.should_process_hour(d_start):
                continue
            for subdir, filename in self.get_file_specs(d_start):
                yield d_start, os.path.join(self.base_url, subdir, filename), os.path.join(self.out_dir, subdir, filename)

    def _download_data(self) -> None:
        LOG.info(f"FixedFileDownloader: Download data. lookback: {self.lookback_hours} lagback: {self.effective_lagback()}")
        final_hour: int = self.effective_lagback() + 1

        for hour in range(self.lookback_hours, self.effective_lagback(), -1):
            d_start: datetime = self.start_time - timedelta(hours=hour)
            self.last_hour = (hour == final_hour)

            if not self.should_process_hour(d_start):
                continue

            for subdir, filename in self.get_file_specs(d_start):
                full_dir: str = os.path.join(self.out_dir, subdir)
                os.makedirs(full_dir, exist_ok=True)
                url: str = os.path.join(self.base_url, subdir, filename)
                out_path: str = os.path.join(full_dir, filename)

                if os.path.isfile(out_path):
                    LOG.info(f"Skipping existing: {out_path}")
                    continue

                self._download_file(url, out_path)

    def _check_data_availability(self) -> None:
        """Check fixed file availability across time range and specs for MRMS and StageIV."""
        LOG.info(f"FixedFileDownloader: Checking data availability. lookback: {self.lookback_hours} lagback: {self.effective_lagback()}")
        total_files_checked: int = 0
        available_files_count: int = 0
        final_hour: int = self.effective_lagback() + 1

        for hour in range(self.lookback_hours, self.effective_lagback(), -1):
            d_start: datetime = self.start_time - timedelta(hours=hour)
            self.last_hour = (hour == final_hour)

            if not self.should_process_hour(d_start):
                continue

            for subdir, filename in self.get_file_specs(d_start):
                total_files_checked += 1
                url: str = os.path.join(self.base_url, subdir, filename)
                LOG.info(f"Checking upstream availability for {subdir}/{filename} via HEAD request")

                if self._check_file_availability(url):
                    available_files_count += 1
                else:
                    LOG.warning(f"Required file missing from server: {url}")

        LOG.info("--- Fixed File Availability Verification Summary ---")
        LOG.info(f"Total files checked: {total_files_checked}")
        LOG.info(f"Total files available on server: {available_files_count}")

        if available_files_count == total_files_checked:
            LOG.info("Verification complete: 100% of requested fixed files are available upstream.")
        else:
            missing_count: int = total_files_checked - available_files_count
            LOG.warning(f"Verification complete: {missing_count} fixed file(s) are currently missing from the server.")
