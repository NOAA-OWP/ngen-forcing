import argparse
import logging
import os
import shutil
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from urllib import error, request

# Use the Error, Warning, and Trapping System Package for logging
import ewts
import requests
from bs4 import BeautifulSoup

LOG = ewts.get_logger(ewts.FORCING_ID)


class ForecastDownloader(ABC):
    """Abstract base class for structured forecast data downloads.

    Supports:
    - Retry-safe downloading
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

    def __init__(
        self,
        out_dir: str,
        start_time: datetime,
        lookback_hours: int | None,
        cleanback_hours: int | None,
        lagback_hours: int | None,
        ens_number: int | None,
        input_horizon=None,
    ) -> None:
        """Initialize downloader with common configuration.

        :param out_dir: Root output directory where files are saved
        :param start_time: time to start forcing extraction
        :param lookback_hours: How many hours back to fetch forecasts
        :param cleanback_hours: How far back to clean old files
        :param lagback_hours: How many hours to lag before starting to fetch
        :param ens_number: Ensemble number to fetch (if applicable)
        :param input_horizon: Maximum forecast hour to download (None = download all available timesteps)
        """
        global LOG
        if hasattr(LOG, "bind"):
            # This is required prior to the first log message for the ewts package
            LOG.bind()
        else:
            # Fallback to default root logger
            logging.basicConfig()
            LOG = logging.getLogger()

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
        self.input_horizon = int(input_horizon / 60)

        # Current hour, rounded to the top of the hour in UTC
        self.d_now = datetime.now(timezone.utc).replace(
            minute=0, second=0, microsecond=0
        )

        # Format ens_number
        self.ens_number = str(self.ens_number).zfill(2)

        # Ensure output directory exists
        os.makedirs(self.out_dir, exist_ok=True)

    def effective_lagback(self):
        """Determine the effective lag back window to avoid downloading data that may not yet be published on the upstream server.

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
        """Create an instance of the subclass using command-line arguments.

        Also prints the parsed arguments for logging/debugging.
        """
        parser = argparse.ArgumentParser()
        parser.add_argument("outDir", type=str, help="Output directory path")
        parser.add_argument(
            "startTime", type=lambda s: datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
        )
        parser.add_argument("--lookBackHours", type=int, default=cls.default_lookback)
        parser.add_argument("--cleanBackHours", type=int, default=cls.default_cleanback)
        parser.add_argument("--lagBackHours", type=int, default=cls.default_lagback)
        parser.add_argument("--ensNumber", type=int, default=None)
        parser.add_argument("--inputHorizon", type=int, default=None)
        args = parser.parse_args()

        print(f"{cls.__name__} args:", vars(args))

        return cls(
            out_dir=args.outDir,
            start_time=args.startTime,
            lookback_hours=args.lookBackHours,
            cleanback_hours=args.cleanBackHours,
            lagback_hours=args.lagBackHours,
            ens_number=args.ensNumber,
            input_horizon=args.inputHorizon,
        )

    def run(self):
        """Cleanup old data, then download new data."""
        self._cleanup_old_data()
        self._download_data()

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
        """Return a list of download targets for a given forecast cycle time.

        This defines what files to download for each cycle timestamp (d_start).
        - For forecast datasets, this might be a list of forecast hours: [0, 1, ..., 18]
        - For radar or QPE datasets, this might be ["Pass1", "Pass2"]
        - If no targets should be downloaded for a given hour, return an empty list []

        This method is called only if should_process_hour(d_current) returns True.
        """
        pass

    def should_process_hour(self, d_start: datetime) -> bool:
        """Determine whether a given forecast cycle hour should be processed.

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
        """Return the output directory for the given forecast cycle datetime."""
        pass

    @abstractmethod
    def build_file_url_and_name(self, d_start, target, ens_number):
        """Return the download URL and filename for a given forecast hour."""
        pass

    #
    # --- Optional subclass overrides ---
    #

    @property
    def recursive_cleanup(self) -> bool:
        """If True, recursively delete leaf directories and prune empty parent directories.

        If False, use default build_output_dir() cleanup per timestamp.
        """
        return False

    def pre_download_hook(self, d_start):
        """Pre-download hook.

        Optional hook called before downloading begins for a specific cycle.

        Use this to scrape directories or cache target lists.
        """
        pass

    def post_download_hook(self, d_start):
        """Post download hook.

        Optional hook called after downloading completes for a specific cycle.
        Use this for logging or post-processing.
        """
        pass

    #
    # --- Internal workflow methods ---
    #

    def _cleanup_old_data(self):
        """Cleanup old data based on the configured lookback and lagback hours.

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
                    LOG.debug(f"Removing old data: {dir_path}")
                    shutil.rmtree(dir_path)

    @staticmethod
    def _remove_dir_and_empty_parents(path, levels: int = 2) -> None:
        """Remove a directory and prunes up to `levels` empty parent directories.

        :param path: Path to the target directory to remove.
        :param levels: Max number of parent levels to prune if empty.
        """
        if os.path.isdir(path):
            LOG.debug(f"Removing directory: {path}")
            shutil.rmtree(path)

            # Prune up to `levels` empty parent directories
            for _ in range(levels):
                path = os.path.dirname(path)
                if os.path.isdir(path) and not os.listdir(path):
                    LOG.debug(f"Removing empty parent directory: {path}")
                    os.rmdir(path)
                else:
                    break

    def _download_data(self) -> None:
        """Download forecast files by iterating over the desired time range and download targets.

        Each timestamp may have one or more targets to process.
        """
        LOG.info(
            f"ForecastDownloader: Download data. lookback: {self.lookback_hours} lagback: {self.effective_lagback()}"
        )
        for hour in range(self.lookback_hours, self.effective_lagback(), -1):
            d_start = self.start_time - timedelta(hours=hour)

            if self.should_process_hour(d_start):
                LOG.debug(f"Processing hour offset: {hour}, timestamp: {d_start}")
            else:
                LOG.debug(f"Skipping hour offset: {hour}, timestamp: {d_start}")
                continue

            output_dir = self.build_output_dir(d_start, self.ens_number)
            os.makedirs(output_dir, exist_ok=True)

            self.pre_download_hook(d_start)

            targets = self.get_download_targets(d_start)

            for target in targets:
                url, filename = self.build_file_url_and_name(
                    d_start, target, self.ens_number
                )
                out_path = os.path.join(output_dir, filename)

                LOG.info(f"Looking for file {out_path}")
                if os.path.isfile(out_path):
                    LOG.info(f"Skipping existing: {out_path}")
                    continue

                self._download_file(url, out_path)

            self.post_download_hook(d_start)

    # noinspection PyMethodMayBeStatic
    def _download_file(self, url, out_path):
        """Download to a unique temporary file, then "publish" it atomically.

        Publishing is done by creating a hard link to the final path.
        This guarantees:
          - no overwriting if another process wins the race
          - no partial files ever appear at the final path
          - atomic, race-proof behavior without lock files
        """
        max_attempts = 10
        interval = 30  # seconds
        attempt = 0

        # Directory and base name for temp file
        out_dir = os.path.dirname(out_path)
        base = os.path.basename(out_path)

        # Format: .<filename>.tmp.<UUID>
        # Hidden temp file tied to the final filename, guaranteed unique
        temp_path = os.path.join(out_dir, f".{base}.tmp.{uuid.uuid4()}")

        while attempt < max_attempts:
            try:
                LOG.info(f"Attempt {attempt + 1}: Downloading {url} -> {temp_path}")
                request.urlretrieve(url, temp_path)

                # ----------------------------------------------------------
                # ATOMIC PUBLISH STEP:
                #
                # os.link(temp_path, out_path) attempts to create a second
                # directory entry (a hard link) pointing to the same inode
                # as the temporary file.
                #
                # This operation is atomic:
                #   - It SUCCEEDS only if 'out_path' does NOT already exist.
                #   - It FAILS with FileExistsError if another process
                #     created the final file first.
                #
                # If it succeeds, both names point to the same inode.
                # We then remove the temp name, leaving only 'out_path'.
                #
                # This effectively acts like a "rename", but:
                #   - it never overwrites existing files
                #   - it is race-safe on Linux
                # ----------------------------------------------------------
                try:
                    os.link(temp_path, out_path)

                    # Give up the temporary name. The underlying file remains,
                    # because 'out_path' now points to the same inode.
                    os.remove(temp_path)

                    LOG.info(f"Download complete: {out_path}")
                    return

                except FileExistsError:
                    # Another process already published the file.
                    LOG.info(
                        f"{out_path} already exists; another process wrote it first. Removing temp."
                    )
                    os.remove(temp_path)
                    return

            except error.HTTPError as e:
                if e.code == 404:
                    # Permanent upstream error — retries won't fix it
                    LOG.error(f"File not found (404): {url} - Stopping retries")
                    return
                # Other HTTP errors may be temporary; allow the retry loop to continue
                LOG.error(f"HTTPError {e.code} while downloading {url}: {e.reason}")

            except error.URLError as e:
                # Network-level transient error; retry is appropriate
                LOG.error(f"URLError while downloading {url}: {e.reason}")

            except Exception as e:
                # Unknown failure; safe to retry
                LOG.error(f"Unexpected error while downloading {url}: {e}")

            finally:
                # If the download step failed, ensure we don't leave stray temp files
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass

            attempt += 1
            time.sleep(interval)

        LOG.error(f"Failed to download after {max_attempts} attempts: {url}")
        return


class FixedFileDownloader(ForecastDownloader, ABC):
    """Subclass for forecast datasets that consist of one or more fixed files per cycle.

    Intended for sources like MRMS and StageIV that have predefined filenames and subdirectories,
    without forecast-hour-based iteration.

    Subclasses must implement:
    - get_file_specs(d_start): returns list of (subdir, filename) for a given timestamp
    """

    def build_file_url_and_name(self, d_start, target):
        """Not used in FixedFileDownloader, since get_file_specs() provides full subdir and filename."""
        raise NotImplementedError("FixedFileDownloader uses get_file_specs() instead.")

    def get_download_targets(self, d_start):
        """Not used in FixedFileDownloader, since targets are provided by get_file_specs()."""
        # Not used in FixedFileDownloader
        return []

    @abstractmethod
    def get_file_specs(self, d_start) -> list[tuple[str, str]]:
        """Return a list of (subdir, filename) tuples for files to be downloaded."""
        pass

    def _download_data(self):
        LOG.info(
            f"FixedFileDownloader: Download data. lookback: {self.lookback_hours} lagback: {self.effective_lagback()}"
        )
        for hour in range(self.lookback_hours, self.effective_lagback(), -1):
            d_start = self.start_time - timedelta(hours=hour)

            if self.should_process_hour(d_start):
                LOG.debug(f"Processing hour offset: {hour}, timestamp: {d_start}")
            else:
                LOG.debug(f"Skipping hour offset: {hour}, timestamp: {d_start}")
                continue

            for subdir, filename in self.get_file_specs(d_start):
                full_dir = os.path.join(self.out_dir, subdir)
                os.makedirs(full_dir, exist_ok=True)
                url = os.path.join(self.base_url, subdir, filename)
                out_path = os.path.join(full_dir, filename)

                if os.path.isfile(out_path):
                    LOG.info(f"Skipping existing: {out_path}")
                    continue

                self._download_file(url, out_path)


class ScrapedFileDownloader(ForecastDownloader, ABC):
    # No longer used, but keeping just in case
    """Subclass for forecast datasets that must scrape an HTML directory to discover files.

    Intended for sources like NBM, where forecast files are published dynamically and filenames
    may vary by cycle.

    Subclasses must implement:
    - get_scrape_url(d_start): returns the remote URL to scrape for a specific timestamp
    - filter_url(url): returns True for valid files to download (e.g., endswith ".hi.grib2")
    """

    @abstractmethod
    def get_scrape_url(self, d_start):
        """Return the URL to scrape for a given forecast cycle timestamp."""
        pass

    @abstractmethod
    def filter_url(self, url: str) -> bool:
        """Return True if the URL points to a valid file to download."""
        pass

    def get_download_targets(self, _):
        """Not used in ScrapedFileDownloader, since targets are discovered via scraping."""
        return [0]  # Satisfy the abstract method; not used for scraping

    def build_file_url_and_name(self, d_start, target):
        """Not used in ScrapedFileDownloader, since URLs and filenames are discovered via scraping."""
        raise NotImplementedError(
            "ScrapedFileDownloader uses scraping logic instead of build_file_url_and_name()."
        )

    def _download_data(self):
        """Download forecast files by scraping directories for the desired time range."""
        LOG.info(
            f"ScrapedFileDownloader: Download data. lookback: {self.lookback_hours} lagback: {self.effective_lagback()}"
        )
        for hour in range(self.lookback_hours, self.effective_lagback(), -1):
            d_start = self.start_time - timedelta(hours=hour)

            if self.should_process_hour(d_start):
                LOG.debug(f"Processing hour offset: {hour}, timestamp: {d_start}")
            else:
                LOG.debug(f"Skipping hour offset: {hour}, timestamp: {d_start}")
                continue

            url = self.get_scrape_url(d_start)
            LOG.info(f"Scraping: {url}")
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
                        LOG.debug(f"Skipping existing: {out_path}")
                        continue

                    self._download_file(full_url, out_path)
