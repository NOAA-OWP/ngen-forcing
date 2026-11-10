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

import requests

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

LOG = logging.getLogger()

#: Seconds to wait for a TCP connection to an upstream server.
CONNECT_TIMEOUT: float = 15.0
#: Seconds to wait between bytes once a response has started.
READ_TIMEOUT: float = 120.0
CHUNK_SIZE: int = 1 << 20


class DownloadError(RuntimeError):
    """A required file could not be retrieved after all attempts."""


def parse_forcing_id(value: str) -> int | str:
    """Parse a forcing identifier: plain digits become an int, anything else (e.g. ``supp12``) stays a str."""
    return int(value) if value.isdigit() else value


def cap_forecast_hours(targets: list[typing.Any], max_hour: int) -> list[typing.Any]:
    """
    Restrict integer forecast-hour targets to those needed for a horizon of ``max_hour`` hours.

    Targets at or below ``max_hour`` are kept. If ``max_hour`` itself is not a target, the first
    target beyond it is also kept so the engine can interpolate up to the horizon. Non-integer
    targets (e.g. NDFD field names) are returned unchanged.
    """
    if not targets or not all(isinstance(t, int) and not isinstance(t, bool) for t in targets):
        return targets
    kept = [t for t in targets if t <= max_hour]
    beyond = [t for t in targets if t > max_hour]
    if beyond and max_hour not in kept:
        kept.append(min(beyond))
    return kept


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
        input_forcing: int | str | None,
        max_download_attempts: int,
        download_attempt_interval: int,
        check_file_availability: int,
        subhourly_avg: bool,
        gfs_extended: bool,
        max_forecast_hour: int | None = None,
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
        self.input_forcing: int | str | None = input_forcing
        self.max_download_attempts: int = max(1, int(max_download_attempts))
        self.download_attempt_interval: int = download_attempt_interval
        self.check_file_availability: int = check_file_availability
        self.subhourly_avg: bool = subhourly_avg
        # Retained for interface compatibility; forecast length is controlled by max_forecast_hour.
        self.gfs_extended: bool = gfs_extended
        self.max_forecast_hour: int | None = max_forecast_hour

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
        parser.add_argument("--inputforcing", type=parse_forcing_id, default=None,
                            help="Forcing identifier (e.g. 6 or supp12); informational only.")
        parser.add_argument("--maxForecastHour", type=int, default=None,
                            help="Download only forecast hours needed to reach this horizon.")
        parser.add_argument("--max_download_attempts", type=int, default=10, help="Number of attempts to download a file.")
        parser.add_argument("--download_attempt_interval", type=int, default=30, help="Wait time (s) between retries.")
        parser.add_argument("--check_file_availability", type=int, default=0, help="Check upstream availability only.")
        parser.add_argument("--subhourly_avg", action="store_true", default=False, help="Flag sub-hourly averaging.")
        parser.add_argument("--gfs_extended", action="store_true", default=False, help="Unused; retained for compatibility.")
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
            max_forecast_hour=args.maxForecastHour,
        )

    @classmethod
    def cli_main(cls) -> None:
        """Command-line entry point: parse arguments, run, and exit non-zero on failure."""
        try:
            ok = cls.from_cli_args().run()
        except DownloadError as exc:
            LOG.error("%s", exc)
            sys.exit(1)
        sys.exit(0 if ok else 1)

    def run(self) -> bool:
        """
        Execute cleanup and downloads, or verify file availability.

        Returns True on success. In availability-check mode, returns False if any file is missing
        upstream. Raises DownloadError if a required download fails.
        """
        if self.check_file_availability == 1:
            return self._check_data_availability()
        self._cleanup_old_data()
        self._download_data()
        return True

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
        """Construct remote file URL and local target filename (or a list of such pairs)."""
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

    # ------------------------------------------------------------------ window helpers

    def _window(self) -> typing.Iterator[tuple[int, datetime]]:
        """Yield (hour offset, cycle time) pairs for the processing window, oldest first."""
        for hour in range(self.lookback_hours, self.effective_lagback(), -1):
            yield hour, self.start_time - timedelta(hours=hour)

    def _protected_cycles(self) -> typing.Iterator[datetime]:
        """Cycle times whose data must survive cleanup: the processing window and anything newer."""
        for hour in range(self.lookback_hours, -1, -1):
            yield self.start_time - timedelta(hours=hour)

    def _targets(self, d_start: datetime) -> list[typing.Any]:
        targets = list(self.get_download_targets(d_start))
        if self.max_forecast_hour is None:
            return targets
        return cap_forecast_hours(targets, self.max_forecast_hour)

    def _url_name_pairs(self, d_start: datetime, target: typing.Any) -> list[tuple[str, str]]:
        result = self.build_file_url_and_name(d_start, target, self.ens_number)
        if isinstance(result, tuple):
            return [result]
        return list(result)

    def iter_remote_files(self) -> typing.Iterator[tuple[datetime, str, str]]:
        """Yield (cycle time, remote URL, local path) for every file the processing window needs."""
        for _, d_start in self._window():
            if not self.should_process_hour(d_start):
                continue
            output_dir = self.build_output_dir(d_start, self.ens_number)
            for target in self._targets(d_start):
                for url, filename in self._url_name_pairs(d_start, target):
                    yield d_start, url, os.path.join(output_dir, filename)

    # ------------------------------------------------------------------ cleanup

    def _cleanup_old_data(self) -> None:
        """Remove output directories of cycles older than the window, never touching data the window needs."""
        if self.cleanback_hours <= self.lookback_hours:
            return
        out_root = os.path.normpath(self.out_dir)
        keep = {
            os.path.normpath(self.build_output_dir(d, self.ens_number))
            for d in self._protected_cycles()
            if self.should_process_hour(d)
        }
        for hour in range(self.cleanback_hours, self.lookback_hours, -1):
            d_start = self.start_time - timedelta(hours=hour)
            if not self.should_process_hour(d_start):
                continue
            path = os.path.normpath(self.build_output_dir(d_start, self.ens_number))
            if path == out_root or path in keep or any(k.startswith(path + os.sep) for k in keep):
                continue
            if self.recursive_cleanup:
                self._remove_dir_and_empty_parents(path, levels=2)
            elif os.path.isdir(path):
                LOG.debug("Removing old data: %s", path)
                shutil.rmtree(path)

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

    # ------------------------------------------------------------------ download

    def _download_data(self) -> None:
        """Download forecast files across desired time horizons and targets."""
        LOG.info(
            f"ForecastDownloader: Download data. lookback: {self.lookback_hours} "
            f"lagback: {self.effective_lagback()}"
        )
        final_hour: int = self.effective_lagback() + 1

        for hour, d_start in self._window():
            self.last_hour = hour == final_hour
            if not self.should_process_hour(d_start):
                LOG.debug(f"Skipping hour offset: {hour}, timestamp: {d_start}")
                continue

            output_dir: str = self.build_output_dir(d_start, self.ens_number)
            os.makedirs(output_dir, exist_ok=True)
            self.pre_download_hook(d_start)
            for target in self._targets(d_start):
                for url, filename in self._url_name_pairs(d_start, target):
                    out_path: str = os.path.join(output_dir, filename)
                    LOG.info(f"Looking for file {out_path}")
                    if os.path.isfile(out_path):
                        LOG.info(f"Skipping existing: {out_path}")
                        continue
                    self._download_file(url, out_path)
            self.post_download_hook(d_start)

    def _fetch(self, url: str, dest_path: str) -> typing.Mapping[str, str]:
        """
        Download ``url`` to ``dest_path`` with retries, verifying the byte count against Content-Length.

        Returns the response headers. Raises DownloadError after the last failed attempt.
        """
        attempts: int = self.max_download_attempts
        last_error = "unknown error"
        for attempt in range(1, attempts + 1):
            try:
                LOG.info("Downloading %s (attempt %d/%d)", url, attempt, attempts)
                with requests.get(url, stream=True, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)) as resp:
                    resp.raise_for_status()
                    expected = resp.headers.get("Content-Length")
                    written = 0
                    with open(dest_path, "wb") as fh:
                        while True:
                            chunk = resp.raw.read(CHUNK_SIZE)
                            if not chunk:
                                break
                            fh.write(chunk)
                            written += len(chunk)
                    if expected is not None and written != int(expected):
                        raise OSError(f"incomplete transfer: received {written} of {expected} bytes")
                    return resp.headers
            except requests.HTTPError as exc:
                last_error = f"HTTP {exc.response.status_code}"
            except requests.Timeout:
                last_error = "timed out"
            except requests.ConnectionError as exc:
                last_error = f"connection error: {exc}"
            except (requests.RequestException, OSError) as exc:
                last_error = str(exc)
            if os.path.exists(dest_path):
                os.remove(dest_path)
            LOG.warning("Attempt %d/%d for %s failed: %s", attempt, attempts, url, last_error)
            if attempt < attempts:
                time.sleep(self.download_attempt_interval)
        raise DownloadError(f"Failed to download {url} after {attempts} attempt(s): {last_error}")

    @staticmethod
    def _publish(temp_path: str, out_path: str) -> None:
        """Move a completed temporary file into place without clobbering a concurrent writer."""
        try:
            os.link(temp_path, out_path)
        except FileExistsError:
            LOG.info("%s already exists; another process wrote it first.", out_path)
        except OSError:
            # Filesystem without hard links: fall back to an atomic rename.
            os.replace(temp_path, out_path)
            return
        os.remove(temp_path)

    def _download_file(self, url: str, out_path: str) -> bool:
        """
        Download remote URL to a temporary path and publish it atomically.

        Returns True when the file was published, False when a tolerated sub-hourly file is missing.
        Raises DownloadError otherwise.
        """
        out_dir: str = os.path.dirname(out_path)
        os.makedirs(out_dir, exist_ok=True)
        temp_path: str = os.path.join(out_dir, f".{os.path.basename(out_path)}.tmp.{uuid.uuid4()}")
        try:
            self._fetch(url, temp_path)
            self._publish(temp_path, out_path)
        except DownloadError as exc:
            if self.last_hour and self.subhourly_avg:
                LOG.warning(
                    "%s. This file is not required for the latest AnA hour when sub-hourly averaging; "
                    "continuing without it.", exc,
                )
                return False
            raise
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        LOG.info("Downloaded %s", out_path)
        return True

    # ------------------------------------------------------------------ availability check

    def _check_data_availability(self) -> bool:
        """Check forecast file availability without fetching content; return True if all files exist."""
        LOG.info(
            f"ForecastDownloader: Checking data availability. lookback: {self.lookback_hours} "
            f"lagback: {self.effective_lagback()}"
        )
        total_files_checked: int = 0
        available_files_count: int = 0
        final_hour: int = self.effective_lagback() + 1

        for hour, d_start in self._window():
            self.last_hour = hour == final_hour
            if not self.should_process_hour(d_start):
                continue

            self.pre_download_hook(d_start)
            all_files_available: bool = True
            for target in self._targets(d_start):
                for url, filename in self._url_name_pairs(d_start, target):
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

        return self._log_availability_summary(total_files_checked, available_files_count)

    @staticmethod
    def _log_availability_summary(total: int, available: int) -> bool:
        LOG.info("--- Forcing Availability Verification Summary ---")
        LOG.info("Total files checked: %d", total)
        LOG.info("Total files available on server: %d", available)
        if available == total:
            LOG.info("Verification complete: all %d requested files are available upstream.", total)
            return True
        LOG.warning("Verification complete: %d file(s) are currently missing from the server.", total - available)
        return False

    def _check_file_availability(self, url: str) -> bool:
        """Ping server using HTTP HEAD request to verify file availability; return False if not found."""
        max_attempts: int = self.max_download_attempts
        interval: int = self.download_attempt_interval
        attempt: int = 0

        while attempt < max_attempts:
            try:
                LOG.info(f"Attempt {attempt + 1}: Pinging {url} for availability...")
                response: requests.Response = requests.head(url, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))

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

        level = logging.WARNING if (self.last_hour and self.subhourly_avg) else logging.ERROR
        LOG.log(level, "Not available after %d attempt(s): %s", max_attempts, url)
        return False


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
        for _, d_start in self._window():
            if not self.should_process_hour(d_start):
                continue
            for subdir, filename in self.get_file_specs(d_start):
                yield d_start, os.path.join(self.base_url, subdir, filename), os.path.join(self.out_dir, subdir, filename)

    def _download_data(self) -> None:
        LOG.info(f"FixedFileDownloader: Download data. lookback: {self.lookback_hours} lagback: {self.effective_lagback()}")
        final_hour: int = self.effective_lagback() + 1

        for hour, d_start in self._window():
            self.last_hour = hour == final_hour

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

    def _check_data_availability(self) -> bool:
        """Check fixed file availability across time range and specs; return True if all files exist."""
        LOG.info(f"FixedFileDownloader: Checking data availability. lookback: {self.lookback_hours} lagback: {self.effective_lagback()}")
        total_files_checked: int = 0
        available_files_count: int = 0
        final_hour: int = self.effective_lagback() + 1

        for hour, d_start in self._window():
            self.last_hour = hour == final_hour

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

        return self._log_availability_summary(total_files_checked, available_files_count)
