import argparse
import atexit
import os
import shutil
import sys
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from urllib import request, error

import requests
from bs4 import BeautifulSoup


class ForecastDownloader(ABC):
    """
    Abstract base class for structured forecast data downloads.

    Supports:
    - Lock file safety
    - Download retry
    - Cleanup of old data
    - CLI support via from_cli_args()
    - Optional forecast-per-cycle or single-file-per-hour strategies

    Subclasses override methods to define behavior for:
    - URL structure
    - File naming
    - Forecast cycle filtering
    """

    default_lookback = 24
    default_cleanback = 240
    default_lagback = 6

    def __init__(self, out_dir, start_time, lookback_hours, cleanback_hours, lagback_hours, ens_number):
        """
        Initialize downloader with common configuration.

        :param out_dir: Root output directory where files are saved
        : param start_time: time to start forcing extraction
        :param lookback_hours: How many hours back to fetch forecasts
        :param cleanback_hours: How far back to clean old files
        :param lagback_hours: How many hours to lag before starting to fetch
        """
        if lookback_hours <= lagback_hours:
            raise ValueError(
                f"Invalid configuration: lookback_hours ({lookback_hours}) must be greater than "
                f"lagback_hours ({lagback_hours}) to allow for a valid processing range."
            )

        self.out_dir = out_dir
        self.start_time = start_time
        self.lookback_hours = lookback_hours
        self.cleanback_hours = cleanback_hours
        self.lagback_hours = lagback_hours
        self.ens_number = ens_number

        # Current hour, rounded to the top of the hour in UTC
        self.d_now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

        # Format ens_number
        self.ens_number = str(self.ens_number).zfill(2)

        # Ensure output directory exists
        os.makedirs(self.out_dir, exist_ok=True)

        # Lockfile used to prevent concurrent runs
        self.lockfile = os.path.join(self.out_dir, f"{self.__class__.__name__}.lck")

    def effective_lagback(self):
        """
        Determines the effective lag back window to avoid downloading data
        that may not yet be published on the upstream server.

        - By default, this returns self.lagback_hours, which is set from the
          command-line argument --lagBackHours (default: 6 hours).
        - However, if a subclass sets self._override_lagback, then that value
          will be used instead — overriding any user-provided CLI input.

        This allows subclasses to enforce a fixed lag window when necessary.

        Currently, this override is only used in `get_conus_HRRR_subhourly`,
        which sets `self._override_lagback = 3` to preserve its original behavior.
        """
        return getattr(self, "_override_lagback", self.lagback_hours)

    #
    # --- Public Interface ---
    #

    @classmethod
    def from_cli_args(cls):
        """
        Create an instance of the subclass using command-line arguments.
        Also prints the parsed arguments for logging/debugging.
        """
        parser = argparse.ArgumentParser()
        parser.add_argument('outDir', type=str, help="Output directory path")
        parser.add_argument('startTime', type=lambda s: datetime.strptime(s, "%Y-%m-%d %H:%M:%S"))
        parser.add_argument('--lookBackHours', type=int, default=cls.default_lookback)
        parser.add_argument('--cleanBackHours', type=int, default=cls.default_cleanback)
        parser.add_argument('--lagBackHours', type=int, default=cls.default_lagback)
        parser.add_argument('--ensNumber', type=int, default=None)
        args = parser.parse_args()

        print(f"{cls.__name__} args:", vars(args))

        return cls(
            out_dir=args.outDir,
            start_time=args.startTime,
            lookback_hours=args.lookBackHours,
            cleanback_hours=args.cleanBackHours,
            lagback_hours=args.lagBackHours,
            ens_number=args.ensNumber
        )

    def run(self):
        """
        Main method that orchestrates lock acquisition, cleanup, and downloading.
        """
        self._acquire_lock()
        try:
            self._cleanup_old_data()
            self._download_data()
        finally:
            self._release_lock()

    #
    # --- Abstract methods to override in subclass ---
    #

    @property
    @abstractmethod
    def base_url(self):
        """Return the base URL for downloading forecast files."""
        pass

    @abstractmethod
    def get_download_targets(self, d_start):
        """
        Return a list of download targets for a given forecast cycle time.

        This defines what files to download for each cycle timestamp (d_start).
        - For forecast datasets, this might be a list of forecast hours: [0, 1, ..., 18]
        - For radar or QPE datasets, this might be ["Pass1", "Pass2"]
        - If no targets should be downloaded for a given hour, return an empty list []

        This method is called only if should_process_hour(d_current) returns True.
        """
        pass

    def should_process_hour(self, d_start: datetime) -> bool:
        """
        Determine whether a given forecast cycle hour should be processed.

        This acts as a fast filter for both downloading and cleanup.
        - Return True if the timestamp is valid for processing (e.g., it's a 6-hour cycle).
        - Return False to skip processing and cleanup for this hour entirely.

        This is helpful for skipping cycles that are available, but we are not interested in them.
        For example, we might only be interested in 00Z, 06Z, 12Z, 18Z, even though cycles are available hourly

        This method is consulted before calling get_download_targets() or build_output_dir().
        """
        return True

    @abstractmethod
    def build_output_dir(self, d_start, ens_number):
        """
        Return the output directory for the given forecast cycle datetime.
        """
        pass

    @abstractmethod
    def build_file_url_and_name(self, d_start, target, ens_number):
        pass

    #
    # --- Optional subclass overrides ---
    #

    @property
    def recursive_cleanup(self) -> bool:
        """
        If True, recursively delete leaf directories and prune empty parent directories.
        If False, use default build_output_dir() cleanup per timestamp.
        """
        return False

    def pre_download_hook(self, d_start):
        """
        Optional hook called before downloading begins for a specific cycle.
        Use this to scrape directories or cache target lists.
        """
        pass

    def post_download_hook(self, d_start):
        """
        Optional hook called after downloading completes for a specific cycle.
        Use this for logging or post-processing.
        """
        pass

    #
    # --- Internal workflow methods ---
    #

    def _acquire_lock(self):
        """
        Prevent multiple concurrent runs by creating a lockfile.
        Register atexit cleanup to ensure the lock is removed when the process exits.
        """
        if os.path.isfile(self.lockfile):
            with open(self.lockfile, 'r') as f:
                pid = f.readline().strip()
            print(f"ERROR: Lock file {self.lockfile} exists. PID: {pid}")
            sys.exit(1)

        with open(self.lockfile, 'w') as f:
            f.write(str(os.getpid()))

        # Automatically clean up the lock file on exit
        atexit.register(self._release_lock)

    def _release_lock(self):
        """
        Safely remove the lock file if it still exists.
        """
        if os.path.exists(self.lockfile):
            try:
                os.remove(self.lockfile)
            except Exception as e:
                print(f"Warning: Failed to remove lockfile: {e}")

    def _cleanup_old_data(self):
        """
        Cleans up old data using either:
        - default timestamp-based cleanup (subdirectory per forecast cycle)
        - recursive directory cleanup with pruning
        """
        for hour in range(self.cleanback_hours, self.lookback_hours, -1):
            d_start = self.start_time - timedelta(hours=hour)
            if not self.should_process_hour(d_start):
                continue

            if self.recursive_cleanup:
                # Recursively remove subdir, parent hour dir, then date dir if empty
                leaf_dir = self.build_output_dir(d_start, self.ens_number)
                self._remove_dir_and_empty_parents(leaf_dir, levels=2)
            else:
                # Default behavior: remove build_output_dir if it exists
                dir_path = self.build_output_dir(d_start, self.ens_number)
                if os.path.isdir(dir_path):
                    print(f"Removing old data: {dir_path}")
                    shutil.rmtree(dir_path)

    @staticmethod
    def _remove_dir_and_empty_parents(path, levels=2):
        """
        Removes a directory and prunes up to `levels` empty parent directories.

        :param path: Path to the target directory to remove.
        :param levels: Max number of parent levels to prune if empty.
        """
        if os.path.isdir(path):
            print(f"Removing directory: {path}")
            shutil.rmtree(path)

            # Prune up to `levels` empty parent directories
            for _ in range(levels):
                path = os.path.dirname(path)
                if os.path.isdir(path) and not os.listdir(path):
                    print(f"Removing empty parent directory: {path}")
                    os.rmdir(path)
                else:
                    break

    def _download_data(self):
        """
        Download forecast files by iterating over the desired time range and download targets.
        Each timestamp may have one or more targets to process.
        """
        for hour in range(self.lookback_hours, self.effective_lagback(), -1):
            d_start = self.start_time - timedelta(hours=hour)

            if self.should_process_hour(d_start):
                print(f"Processing hour offset: {hour}, timestamp: {d_start}")
            else:
                print(f"Skipping hour offset: {hour}, timestamp: {d_start}")
                continue

            output_dir = self.build_output_dir(d_start, self.ens_number)
            os.makedirs(output_dir, exist_ok=True)

            self.pre_download_hook(d_start)

            targets = self.get_download_targets(d_start)
            for target in targets:
                url, filename = self.build_file_url_and_name(d_start, target, self.ens_number)
                out_path = os.path.join(output_dir, filename)

                if os.path.isfile(out_path):
                    print(f"Skipping existing: {out_path}")
                    continue

                self._download_file(url, out_path)

            self.post_download_hook(d_start)

    # noinspection PyMethodMayBeStatic
    def _download_file(self, url, out_path):
        """
        Attempt to download a file from the URL with retry logic.
        Retries up to 10 times with a 30-second interval between attempts.
        """
        max_attempts = 10
        interval = 30  # seconds
        attempt = 0

        while attempt < max_attempts:
            try:
                print(f"Attempt {attempt + 1}: Downloading {url}")
                request.urlretrieve(url, out_path)
                print(f"Download complete: {out_path}")
                return
            except error.HTTPError as e:
                if e.code == 404:
                    print(f"File not found (404): {url} - Stopping retries")
                    return False
                print(f"HTTPError {e.code} while downloading {url}: {e.reason}")
            except error.URLError as e:
                print(f"URLError while downloading {url}: {e.reason}")
            except Exception as e:
                print(f"Unexpected error while downloading {url}: {e}")

            attempt += 1
            time.sleep(interval)

        print(f"Failed to download after {max_attempts} attempts: {url}")


class FixedFileDownloader(ForecastDownloader, ABC):
    """
    Subclass for forecast datasets that consist of one or more fixed files per cycle.

    Intended for sources like MRMS and StageIV that have predefined filenames and subdirectories,
    without forecast-hour-based iteration.

    Subclasses must implement:
    - get_file_specs(d_start): returns list of (subdir, filename) for a given timestamp
    """

    def build_file_url_and_name(self, d_start, target):
        raise NotImplementedError("FixedFileDownloader uses get_file_specs() instead.")

    def get_download_targets(self, d_start):
        # Not used in FixedFileDownloader
        return []

    @abstractmethod
    def get_file_specs(self, d_start) -> list[tuple[str, str]]:
        """
        Return a list of (subdir, filename) tuples for files to be downloaded
        """
        pass

    def _download_data(self):
        for hour in range(self.lookback_hours, self.effective_lagback(), -1):
            d_start = self.start_time - timedelta(hours=hour)

            if self.should_process_hour(d_start):
                print(f"Processing hour offset: {hour}, timestamp: {d_start}")
            else:
                print(f"Skipping hour offset: {hour}, timestamp: {d_start}")
                continue

            for subdir, filename in self.get_file_specs(d_start):
                full_dir = os.path.join(self.out_dir, subdir)
                os.makedirs(full_dir, exist_ok=True)
                url = os.path.join(self.base_url, subdir, filename)
                out_path = os.path.join(full_dir, filename)

                if os.path.isfile(out_path):
                    print(f"Skipping existing: {out_path}")
                    continue

                self._download_file(url, out_path)


class ScrapedFileDownloader(ForecastDownloader, ABC):
    # No longer used, but keeping just in case
    """
    Subclass for forecast datasets that must scrape an HTML directory to discover files.

    Intended for sources like NBM, where forecast files are published dynamically and filenames
    may vary by cycle.

    Subclasses must implement:
    - get_scrape_url(d_start): returns the remote URL to scrape for a specific timestamp
    - filter_url(url): returns True for valid files to download (e.g., endswith ".hi.grib2")
    """

    @abstractmethod
    def get_scrape_url(self, d_start):
        pass

    @abstractmethod
    def filter_url(self, url: str) -> bool:
        pass

    def get_download_targets(self, _):
        return [0]  # Satisfy the abstract method; not used for scraping

    def build_file_url_and_name(self, d_start, target):
        raise NotImplementedError("ScrapedFileDownloader uses scraping logic instead of build_file_url_and_name().")

    def _download_data(self):
        for hour in range(self.lookback_hours, self.effective_lagback(), -1):
            d_start = self.start_time - timedelta(hours=hour)

            if self.should_process_hour(d_start):
                print(f"Processing hour offset: {hour}, timestamp: {d_start}")
            else:
                print(f"Skipping hour offset: {hour}, timestamp: {d_start}")
                continue

            url = self.get_scrape_url(d_start)
            print(f"Scraping: {url}")
            output_dir = self.build_output_dir(d_start)
            os.makedirs(output_dir, exist_ok=True)

            html = requests.get(url).text
            soup = BeautifulSoup(html, "html.parser")
            for a in soup.find_all("a"):
                href = a.get("href", "")
                full_url = os.path.join(url, href)
                if self.filter_url(full_url):
                    out_path = os.path.join(output_dir, os.path.basename(full_url))

                    if os.path.isfile(out_path):
                        print(f"Skipping existing: {out_path}")
                        continue

                    self._download_file(full_url, out_path)
