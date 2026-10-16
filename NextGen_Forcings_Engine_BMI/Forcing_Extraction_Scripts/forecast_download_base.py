import argparse
import logging
import os
import sys
import shutil
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from urllib import request, error

import requests
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout, # Forces it to use standard output instead of standard error
    format='%(asctime)s - %(levelname)s - %(message)s'
)

LOG = logging.getLogger()


class ForecastDownloader(ABC):
    """
    Abstract base class for structured forecast data downloads.

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

    def __init__(self, out_dir, start_time, lookback_hours, cleanback_hours, lagback_hours, ens_number, input_forcing, max_download_attempts, download_attempt_interval, check_file_availability, subhourly_avg, gfs_extended):
        """
        Initialize downloader with common configuration.

        :param out_dir: Root output directory where files are saved
        : param start_time: time to start forcing extraction
        :param lookback_hours: How many hours back to fetch forecasts
        :param cleanback_hours: How far back to clean old files
        :param lagback_hours: How many hours to lag before starting to fetch
        :param ens_number: The ensemble number for a given operational model
        :param input_forcing: The integer based value of a specific operational model in forcing engine
        :param max_download_attempts: How many attempts do you want make for downloading a forcing file
        :param download_attempt_interval: The number of seconds you want to wait between each attempt to download a forcing file after a failure
        :param check_file_availability: An integer value that indicates whether or not to defer to just checking file availability rather than downloading files
        :param subhourly_avg: A boolean flag that indicates the forcing data product is being used for sub-hourly averaging, which flags a different check for AnA
        :param gfs_extended: A boolean flag to indicate whether or not we need to download GFS 10-day or 16-day forecast based on operational configuration of interest
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
        self.input_forcing = input_forcing
        self.max_download_attempts = max_download_attempts
        self.download_attempt_interval = download_attempt_interval
        self.check_file_availability = check_file_availability
        self.subhourly_avg = subhourly_avg
        self.gfs_extended = gfs_extended
        
        # Self catching mechanism to infer last hour of user download request
        self.last_hour = None

        # Current hour, rounded to the top of the hour in UTC
        self.d_now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

        # Format ens_number
        self.ens_number = str(self.ens_number).zfill(2)

        # Ensure output directory exists
        os.makedirs(self.out_dir, exist_ok=True)

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
        parser.add_argument('--inputforcing', type=int)
        parser.add_argument('--max_download_attempts', type=int, default=10,help="Number of attempts to download a file.")
        parser.add_argument('--download_attempt_interval', type=int, default=30,help="How many seconds do you wait until you attempt to retry downloading a file that failed.")
        parser.add_argument('--check_file_availability', type=int, default=0, help="Bypass file downloading and just check to see if files are available to download. To implement specify 1, or omit you specify 0. Default is 0.")        
        parser.add_argument('--subhourly_avg',action="store_true",default=False,help="Indicate that the sub-hourly averaging method is being used for AnA cycle. (default: False)")
        parser.add_argument('--gfs_extended',action="store_true",default=False,help="Indicate GFS 10-day vs 16-day downloading based on operational configuration of interest. (default: False)")
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
            subhourly_avg=subhourly_avg,
            gfs_extended=gfs_extended
        )

    def run(self):
        """
        Main method that orchestrates lock acquisition, cleanup, and downloading OR just check availability of forcing files.
        """
        if(self.check_file_availability==1):
            self._check_data_availability()
        else:
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
                    LOG.debug(f"Removing old data: {dir_path}")
                    shutil.rmtree(dir_path)

    @staticmethod
    def _remove_dir_and_empty_parents(path, levels=2):
        """
        Removes a directory and prunes up to `levels` empty parent directories.

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

    def _download_data(self):
        """
        Download forecast files by iterating over the desired time range and download targets.
        Each timestamp may have one or more targets to process.
        """
        LOG.info(f"ForecastDownloader: Download data. lookback: {self.lookback_hours} lagback: {self.effective_lagback()}")
        # Initalize the last hour of the operational cycle
        final_hour = self.effective_lagback() + 1
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
            # Set last hour flag based on current hour of cycle
            if hour == final_hour:
                self.last_hour = True
            else:
                self.last_hour = False

            # Flag for special RAP case where we need multiple file types
            if self.input_forcing == 6:
                for target in targets:
                    file_list = self.build_file_url_and_name(d_start, target, self.ens_number)
                    for url, filename in file_list:
                        out_path = os.path.join(output_dir, filename)

                        LOG.info(f"Looking for file {out_path}")
                        if os.path.isfile(out_path):
                            LOG.info(f"Skipping existing: {out_path}")
                            continue

                        self._download_file(url, out_path)

                self.post_download_hook(d_start)
            else:
                for target in targets:
                    url, filename = self.build_file_url_and_name(d_start, target, self.ens_number)
                    out_path = os.path.join(output_dir, filename)

                    LOG.info(f"Looking for file {out_path}")
                    if os.path.isfile(out_path):
                        LOG.info(f"Skipping existing: {out_path}")
                        continue

                    self._download_file(url, out_path)

                self.post_download_hook(d_start)

    # noinspection PyMethodMayBeStatic
    def _download_file(self, url, out_path):
        """
        Download to a unique temporary file, then "publish" it atomically.
        Publishing is done by creating a hard link to the final path.
        This guarantees:
          - no overwriting if another process wins the race
          - no partial files ever appear at the final path
          - atomic, race-proof behavior without lock files
        """
        max_attempts = self.max_download_attempts
        interval = self.download_attempt_interval  # seconds
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
                    LOG.info(f"{out_path} already exists; another process wrote it first. Removing temp.")
                    os.remove(temp_path)
                    return

            except error.HTTPError as e:
                # HTTP errors may be temporary; allow the retry loop to continue as user requested
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

        # If we're on the last hour and sub-hourly AnA operational scheme is used
        # then do throw an kill switch as this last file is not absolutely necessary
        if(self.last_hour and self.subhourly_avg):
            LOG.warning(f"Failed to download after {max_attempts} attempts: {url}")
            LOG.warning(f"This MRMS file is not required to fill in the latest AnA cycle hour for sub-hourly averaging method. Will use HRRR instead.")
        else:
            LOG.error(f"Failed to download after {max_attempts} attempts: {url}")
            LOG.critical(f"Script terminating with exit code 1 due to 404 error on {url}")
            os._exit(1)
        
    def _check_data_availability(self):
        """
        Check forecast file availability across the desired time range and targets.
        Tracks and logs the total count of available files discovered.
        """
        LOG.info(f"ForecastDownloader: Checking data availability. lookback: {self.lookback_hours} lagback: {self.effective_lagback()}")

        # Global counters for the entire run
        total_files_checked = 0
        available_files_count = 0

        # Initalize the last hour of the operational cycle
        final_hour = self.effective_lagback() + 1

        for hour in range(self.lookback_hours, self.effective_lagback(), -1):
            d_start = self.start_time - timedelta(hours=hour)

            if self.should_process_hour(d_start):
                LOG.debug(f"Processing hour offset: {hour}, timestamp: {d_start}")
            else:
                LOG.debug(f"Skipping hour offset: {hour}, timestamp: {d_start}")
                continue


            self.pre_download_hook(d_start)

            targets = self.get_download_targets(d_start)

            # Set last hour flag based on current hour of cycle
            if hour == final_hour:
                self.last_hour = True
            else:
                self.last_hour = False

            all_files_available = True  # Track status for this specific timestamp block

            # Special RAP case where build_file_url_and_name returns a list of (url, filename)
            if self.input_forcing == 6:
                for target in targets:
                    file_list = self.build_file_url_and_name(d_start, target, self.ens_number)

                    for url, filename in file_list:
                        total_files_checked += 1
                        LOG.info(f"Checking upstream availability for {filename} via HEAD request")

                        if self._check_file_availability(url):
                            available_files_count += 1
                        else:
                            all_files_available = False
                            LOG.warning(f"Required file missing from server: {url}")

            # Standard models returning a single (url, filename) tuple
            else:
                for target in targets:
                    url, filename = self.build_file_url_and_name(d_start, target, self.ens_number)
                    total_files_checked += 1
                    LOG.info(f"Checking upstream availability for {filename} via HEAD request")

                    if self._check_file_availability(url):
                        available_files_count += 1
                    else:
                        all_files_available = False
                        LOG.warning(f"Required file missing from server: {url}")

            # Post-download hook handling based on timestamp block completeness
            if all_files_available:
                LOG.info(f"All required forcing files are available for timestamp: {d_start}")
                self.post_download_hook(d_start)
            else:
                LOG.error(f"One or more forcing files missing for timestamp: {d_start}. Post-download hook skipped.")

        # Final summary metrics logged after the outer loop finishes completely
        LOG.info(f"--- Forcing Availability Verification Summary ---")
        LOG.info(f"Total files checked: {total_files_checked}")
        LOG.info(f"Total files available on server: {available_files_count}")

        if available_files_count == total_files_checked:
            LOG.info("Verification complete: 100% of requested files are available upstream.")
        else:
            missing_count = total_files_checked - available_files_count
            LOG.warning(f"Verification complete: {missing_count} file(s) are currently missing from the server.")

    # noinspection PyMethodMayBeStatic
    def _check_file_availability(self, url):
        """
        Ping the server using an HTTP HEAD request to check if the file is available.

        Returns:
            bool: True if the file exists (HTTP 200), False if it does not (404) or fails.
        """
        max_attempts = self.max_download_attempts
        interval = self.download_attempt_interval  # seconds
        attempt = 0

        while attempt < max_attempts:
            try:
                LOG.info(f"Attempt {attempt + 1}: Pinging {url} for availability...")

                # Send an HTTP HEAD request with a reasonable timeout
                response = requests.head(url, timeout=10)

                # If status code is 200, the file is ready
                if response.status_code == requests.codes.ok:  # status_code == 200
                    LOG.info(f"File is available: {url}")
                    return True

                elif response.status_code == 404:
                    LOG.warning(f"File not found (404): {url}")
                    
                else:
                    LOG.error(f"Server returned status code {response.status_code} while pinging {url}")

            except requests.exceptions.Timeout:
                # Handle connection or read timeouts gracefully
                LOG.error(f"Timeout occurred while trying to connect to {url}")

            except requests.exceptions.ConnectionError:
                # Handle DNS failures, refused connections, or network drops
                LOG.error(f"ConnectionError (Network issue) while pinging {url}")

            except requests.exceptions.RequestException as e:
                # Catch-all for any other anomalous Requests-related errors
                LOG.error(f"An error occurred with the request: {e}")

            attempt += 1

            # Only sleep if we have remaining attempts left
            if attempt < max_attempts:
                time.sleep(interval)

        # If we're on the last hour and sub-hourly AnA operational scheme is used
        # then do throw an kill switch as this last file is not absolutely necessary
        if(self.last_hour and self.subhourly_avg):
            LOG.warning(f"Failed to download after {max_attempts} attempts: {url}")
            LOG.warning(f"This MRMS file is not required to fill in the latest AnA cycle hour for sub-hourly averaging method. Will use HRRR instead.")
        else:
            LOG.error(f"Failed to confirm file availability after {max_attempts} attempts: {url}")
            LOG.critical(f"Script terminating with exit code 1 due to 404 error on {url}")
            os._exit(1)
        
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
        LOG.info(f"FixedFileDownloader: Download data. lookback: {self.lookback_hours} lagback: {self.effective_lagback()}")
        # Initalize the last hour of the operational cycle
        final_hour = self.effective_lagback() + 1
        for hour in range(self.lookback_hours, self.effective_lagback(), -1):
            d_start = self.start_time - timedelta(hours=hour)

            # Set last hour flag based on current hour of cycle
            if hour == final_hour:
                self.last_hour = True
            else:
                self.last_hour = False

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

    def _check_data_availability(self):
        """
        Check fixed file availability across the desired time range and specifications.
        Instead of downloading, this validates if the files are available on the remote server.
        """
        LOG.info(f"FixedFileDownloader: Checking data availability. lookback: {self.lookback_hours} lagback: {self.effective_lagback()}")

        # Initialize global counters for the entire verification run
        total_files_checked = 0
        available_files_count = 0

        # Initalize the last hour of the operational cycle
        final_hour = self.effective_lagback() + 1
        for hour in range(self.lookback_hours, self.effective_lagback(), -1):
            d_start = self.start_time - timedelta(hours=hour)

            # Set last hour flag based on current hour of cycle
            if hour == final_hour:
                self.last_hour = True
            else:
                self.last_hour = False

            if self.should_process_hour(d_start):
                LOG.debug(f"Processing hour offset: {hour}, timestamp: {d_start}")
            else:
                LOG.debug(f"Skipping hour offset: {hour}, timestamp: {d_start}")
                continue

            # Loop through the files defined for this timestamp
            for subdir, filename in self.get_file_specs(d_start):
                total_files_checked += 1

                # Construct the upstream URL to ping
                url = os.path.join(self.base_url, subdir, filename)

                LOG.info(f"Checking upstream availability for {subdir}/{filename} via HEAD request")

                # Use the requests-based ping method to check server presence
                if self._check_file_availability(url):
                    available_files_count += 1
                else:
                    LOG.warning(f"Required file missing from server: {url}")

        # Final summary metrics logged after scanning all hours and file specs
        LOG.info(f"--- Fixed File Availability Verification Summary ---")
        LOG.info(f"Total files checked: {total_files_checked}")
        LOG.info(f"Total files available on server: {available_files_count}")

        if available_files_count == total_files_checked:
            LOG.info("Verification complete: 100% of requested fixed files are available upstream.")
        else:
            missing_count = total_files_checked - available_files_count
            LOG.warning(f"Verification complete: {missing_count} fixed file(s) are currently missing from the server.")
            
