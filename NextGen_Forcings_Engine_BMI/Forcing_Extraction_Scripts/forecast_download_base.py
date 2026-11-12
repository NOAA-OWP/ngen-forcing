from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import sys
import time
import typing
import uuid
import zlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta

import requests
import urllib3

LOG = logging.getLogger(__name__)

#: Seconds to wait for a TCP connection to an upstream server.
CONNECT_TIMEOUT: float = 15.0
#: Seconds to wait between bytes once a response has started.
READ_TIMEOUT: float = 120.0
CHUNK_SIZE: int = 1 << 20
#: Default number of attempts per file, and seconds between attempts.
DEFAULT_MAX_ATTEMPTS: int = 10
DEFAULT_ATTEMPT_INTERVAL: int = 10
#: Bytes read from the start of a file to identify its content.
SNIFF_BYTES: int = 4096
#: Ask servers and proxies for the stored bytes, never a transfer-compressed form.
#:
#: This keeps Content-Length, MD5 ETags and byte ranges referring to the stored object. On
#: 2026-11-10 it cost nothing: offered gzip, the S3 buckets returned the stored bytes and NOMADS
#: compressed only directory listings. gzip would shrink some products substantially (Stage IV and
#: NDFD ~75%, NBM ~34%, GFS/HRRR ~15%), so check_server_behavior.py reports whether servers would
#: compress; if they start to, accepting gzip with decoding (and without MD5/resume) is worth it.
REQUEST_HEADERS: dict[str, str] = {"Accept-Encoding": "identity"}


#: NAM and HiResW (except Guam) ended with the RRFS implementation on 2026-11-03; the last NAM cycle
#: in the S3 archive is 06Z that day. Earlier cycles are still fetched, for retrospective runs.
NAM_HIRESW_RETIRED_AFTER = datetime(2026, 11, 3, 6)
NAM_HIRESW_RETIREMENT_NOTE = (
    "NAM and HiResW were retired on 2026-11-03 and replaced by RRFS "
    "(NWS Service Change Notices 26-47 and 26-48)"
)


class DownloadError(RuntimeError):
    """A required file could not be retrieved after all attempts."""


class ServerInconsistency(OSError):
    """A server response contradicts its own metadata (length, checksum, ranges, content type)."""


def describe_request_error(exc: BaseException) -> str:
    """A short description of why an HTTP request or transfer failed."""
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return f"HTTP {exc.response.status_code}"
    if isinstance(exc, requests.Timeout):
        return "timed out"
    if isinstance(exc, requests.ConnectionError):
        return f"connection error: {exc}"
    if isinstance(exc, ServerInconsistency):
        return f"server response inconsistent with its metadata: {exc}"
    return f"{type(exc).__name__}: {exc}"


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging for command-line entry points."""
    logging.basicConfig(
        level=level,
        stream=sys.stdout,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def parse_forcing_id(value: str) -> int | str:
    """Parse a forcing identifier: plain digits become an int, anything else (e.g. ``supp12``) stays a str."""
    return int(value) if value.isdigit() else value


_PLAIN_MD5_ETAG = re.compile(r'^"?([0-9a-f]{32})"?$')


def plain_md5_etag(etag: str | None) -> str | None:
    """Return the MD5 hex digest an ETag stands for, or None for multipart/opaque ETags."""
    match = _PLAIN_MD5_ETAG.match(etag or "")
    return match.group(1) if match else None


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


def content_signature_problem(name: str, head: bytes, require_edition2: bool = False) -> str:
    """
    Check the first bytes of a file against what its name (or URL) says it is.

    ``.gz`` files must be gzip whose decompressed start is checked in turn; ``.grib2``, ``.grb2``
    and ``.bin`` files must contain a GRIB marker (operational bulletins carry a WMO header before
    it). Returns '' if consistent, else a description of what was found.
    """
    lower = name.rsplit("/", 1)[-1].split("?", 1)[0].lower()
    if lower.endswith(".gz"):
        if head[:2] != b"\x1f\x8b":
            return f"named .gz but starts with {head[:8]!r}"
        try:
            head = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(head)
        except zlib.error as exc:
            return f"gzip data is undecodable: {exc}"
        lower = lower[:-3]
    if lower.endswith((".grib2", ".grb2", ".bin")):
        offset = head.find(b"GRIB")
        if offset < 0:
            return f"no GRIB marker in the first {len(head)} bytes (starts {head[:8]!r})"
        if require_edition2 and len(head) > offset + 7 and head[offset + 7] != 2:
            return f"GRIB edition {head[offset + 7]}, expected 2"
    return ""


class ForecastDownloader(ABC):
    """Abstract base class for structured forecast data downloads."""

    default_lookback: int = 24
    default_cleanback: int = 240
    default_lagback: int = 6

    #: Last cycle the upstream product exists for, if it has been retired; later cycles fail at once.
    retired_after: typing.ClassVar[datetime | None] = None
    #: Why the product ended, with a reference; included in the error for later cycles.
    retirement_note: typing.ClassVar[str] = ""

    def __init__(
        self,
        out_dir: str | os.PathLike,
        start_time: datetime,
        lookback_hours: int,
        cleanback_hours: int = 0,
        lagback_hours: int = 0,
        ens_number: int | str | None = None,
        input_forcing: int | str | None = None,
        max_download_attempts: int = DEFAULT_MAX_ATTEMPTS,
        download_attempt_interval: int = DEFAULT_ATTEMPT_INTERVAL,
        check_file_availability: int = 0,
        subhourly_avg: bool = False,
        gfs_extended: bool = False,
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
        self.ens_number: str = str(ens_number).zfill(2) if ens_number not in (None, "") else "00"
        self.input_forcing: int | str | None = input_forcing
        self.max_download_attempts: int = max(1, int(max_download_attempts))
        self.download_attempt_interval: int = download_attempt_interval
        self.check_file_availability: int = check_file_availability
        self.subhourly_avg: bool = subhourly_avg
        # Retained for interface compatibility; forecast length is controlled by max_forecast_hour.
        self.gfs_extended: bool = gfs_extended
        self.max_forecast_hour: int | None = max_forecast_hour

        self.last_hour: bool | None = None

    def effective_lagback(self) -> int:
        """Return the lag-back (in hours) that ends the processing window; subclasses may override."""
        return self.lagback_hours

    @classmethod
    def from_cli_args(
        cls,
        desc: str | None = None,
        argv: typing.Sequence[str] | None = None,
    ) -> typing.Self:
        """Create an instance of the subclass using command-line arguments."""
        parser = argparse.ArgumentParser(description=desc or cls.__doc__)
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
        parser.add_argument("--max_download_attempts", type=int, default=DEFAULT_MAX_ATTEMPTS,
                            help="Number of attempts to download a file.")
        parser.add_argument("--download_attempt_interval", type=int, default=DEFAULT_ATTEMPT_INTERVAL,
                            help="Wait time (s) between retries.")
        parser.add_argument("--check_file_availability", type=int, default=0, help="Check upstream availability only.")
        parser.add_argument("--subhourly_avg", action="store_true", default=False, help="Flag sub-hourly averaging.")
        parser.add_argument("--gfs_extended", action="store_true", default=False, help="Unused; retained for compatibility.")
        args = parser.parse_args(argv)
        LOG.info("%s args: %s", cls.__name__, vars(args))

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
    def cli_main(cls, desc: str | None = None) -> None:
        """Command-line entry point: parse arguments, run, and exit non-zero on failure."""
        configure_logging()
        try:
            ok = cls.from_cli_args(desc).run()
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

    def _cycle_files(self, d_start: datetime) -> list[tuple[str, str]]:
        """(remote URL, local path) for every file one cycle needs."""
        output_dir = self.build_output_dir(d_start, self.ens_number)
        return [
            (url, os.path.join(output_dir, filename))
            for target in self._targets(d_start)
            for url, filename in self._url_name_pairs(d_start, target)
        ]

    def _cycles(self) -> typing.Iterator[tuple[datetime, list[tuple[str, str]]]]:
        """Yield (cycle time, files) for each cycle in the window that has files; sets ``last_hour``."""
        final_hour: int = self.effective_lagback() + 1
        for hour, d_start in self._window():
            self.last_hour = hour == final_hour
            if not self.should_process_hour(d_start):
                LOG.debug("Skipping hour offset: %s, timestamp: %s", hour, d_start)
                continue
            files = self._cycle_files(d_start)
            if files:
                yield d_start, files

    def _part_urls(self, url: str, out_path: str) -> list[str]:
        """Remote parts concatenated, in order, into ``out_path``; normally just ``url``."""
        return [url]

    def _check_part(self, url: str, headers: typing.Mapping[str, str]) -> None:
        """Hook to reject a downloaded part from its response headers by raising DownloadError."""

    def iter_remote_files(self) -> typing.Iterator[tuple[datetime, str, str]]:
        """Yield (cycle time, remote URL, local path) for every remote part the processing window needs."""
        for d_start, files in self._cycles():
            for url, path in files:
                for part in self._part_urls(url, path):
                    yield d_start, part, path

    def retirement_problem(self, d_start: datetime) -> str | None:
        """Why no data can exist for cycle ``d_start``, if the product was retired before it."""
        if self.retired_after is None or d_start <= self.retired_after:
            return None
        return (f"{type(self).__name__}: no data after the {self.retired_after:%Y-%m-%d %HZ} cycle "
                f"({self.retirement_note}); cannot supply {d_start:%Y-%m-%d %HZ}")

    def _warn_nothing_processed(self) -> None:
        first = self.start_time - timedelta(hours=self.lookback_hours)
        last = self.start_time - timedelta(hours=self.effective_lagback() + 1)
        LOG.warning(
            "%s: no cycle between %s and %s matches this product's cycle schedule; nothing was downloaded.",
            type(self).__name__, first, last,
        )

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
            LOG.debug("Removing directory: %s", path)
            shutil.rmtree(path)

            current_path: str = path
            for _ in range(levels):
                current_path = os.path.dirname(current_path)
                if os.path.isdir(current_path) and not os.listdir(current_path):
                    LOG.debug("Removing empty parent directory: %s", current_path)
                    os.rmdir(current_path)
                else:
                    break

    # ------------------------------------------------------------------ download

    def _download_data(self) -> None:
        """Download forecast files across desired time horizons and targets."""
        LOG.info(
            "%s: downloading. lookback: %s lagback: %s",
            type(self).__name__, self.lookback_hours, self.effective_lagback(),
        )
        processed = 0
        for d_start, files in self._cycles():
            processed += 1
            problem = self.retirement_problem(d_start)
            if problem:
                raise DownloadError(problem)
            self.pre_download_hook(d_start)
            for url, out_path in files:
                if os.path.isfile(out_path):
                    LOG.debug("Skipping existing: %s", out_path)
                    continue
                self._download_file(url, out_path)
            self.post_download_hook(d_start)

        if processed == 0:
            self._warn_nothing_processed()

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
                with requests.get(url, stream=True, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
                                  headers=REQUEST_HEADERS) as resp:
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
                        raise ServerInconsistency(f"received {written} bytes but Content-Length is {expected}")
                    with open(dest_path, "rb") as fh:
                        problem = content_signature_problem(url, fh.read(SNIFF_BYTES))
                    if problem:
                        raise ServerInconsistency(f"content does not match {url.rsplit('/', 1)[-1]}: {problem}")
                    return resp.headers
            except (requests.RequestException, urllib3.exceptions.HTTPError, OSError) as exc:
                # Errors while streaming the body come from urllib3, not requests.
                last_error = describe_request_error(exc)
                if isinstance(exc, ServerInconsistency):
                    LOG.error("%s: %s", url, last_error)
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
        Download a file (or all its parts) to temporary paths and publish it atomically.

        A multi-part file is published only when every part downloaded. Returns True when the file
        was published, False when a tolerated sub-hourly file is missing. Raises DownloadError
        otherwise.
        """
        out_dir: str = os.path.dirname(out_path)
        os.makedirs(out_dir, exist_ok=True)
        parts = self._part_urls(url, out_path)
        name, token = os.path.basename(out_path), uuid.uuid4()
        temp_path: str = os.path.join(out_dir, f".{name}.tmp.{token}")
        part_paths = [temp_path] if len(parts) == 1 else [
            os.path.join(out_dir, f".{name}.part{i}.{token}") for i in range(len(parts))]
        try:
            for part_url, part_path in zip(parts, part_paths):
                self._check_part(part_url, self._fetch(part_url, part_path))
            if len(parts) > 1:
                with open(temp_path, "wb") as outfile:
                    for part_path in part_paths:
                        with open(part_path, "rb") as infile:
                            shutil.copyfileobj(infile, outfile)
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
            for path in {temp_path, *part_paths}:
                if os.path.exists(path):
                    os.remove(path)
        LOG.info("Downloaded %s", out_path)
        return True

    # ------------------------------------------------------------------ availability check

    def _check_data_availability(self) -> bool:
        """Check forecast file availability without fetching content; return True if all files exist."""
        LOG.info(
            "%s: checking data availability. lookback: %s lagback: %s",
            type(self).__name__, self.lookback_hours, self.effective_lagback(),
        )
        total_files_checked: int = 0
        available_files_count: int = 0
        for d_start, files in self._cycles():
            problem = self.retirement_problem(d_start)
            if problem:
                LOG.error("%s", problem)
                total_files_checked += sum(len(self._part_urls(u, p)) for u, p in files)
                continue
            self.pre_download_hook(d_start)
            all_files_available: bool = True
            for url, path in files:
                for part in self._part_urls(url, path):
                    total_files_checked += 1
                    if self._check_file_availability(part):
                        available_files_count += 1
                    else:
                        all_files_available = False

            if all_files_available:
                LOG.info("All required forcing files are available for timestamp: %s", d_start)
                self.post_download_hook(d_start)
            else:
                LOG.error("One or more forcing files missing for timestamp: %s.", d_start)

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
        attempts: int = self.max_download_attempts
        last_error = "unknown error"
        for attempt in range(1, attempts + 1):
            try:
                response: requests.Response = requests.head(url, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
                                                             headers=REQUEST_HEADERS)
                if response.status_code == requests.codes.ok:
                    LOG.debug("File is available: %s", url)
                    return True
                last_error = f"HTTP {response.status_code}"
            except requests.RequestException as exc:
                last_error = describe_request_error(exc)
            LOG.debug("Availability attempt %d/%d for %s failed: %s", attempt, attempts, url, last_error)
            if attempt < attempts:
                time.sleep(self.download_attempt_interval)

        level = logging.WARNING if (self.last_hour and self.subhourly_avg) else logging.ERROR
        LOG.log(level, "Not available after %d attempt(s): %s (%s)", attempts, url, last_error)
        return False


@dataclass(frozen=True)
class FileSpec:
    """One fixed file: its directory on the server, its name, and its directory under ``out_dir``."""

    remote_dir: str
    filename: str
    local_dir: str = ""


class FixedFileDownloader(ForecastDownloader, ABC):
    """Subclass for MRMS and StageIV datasets with predefined filenames and subdirectories."""

    def build_file_url_and_name(self, d_start: datetime, target: typing.Any, ens_number: str) -> tuple[str, str]:
        raise NotImplementedError("FixedFileDownloader uses get_file_specs() instead.")

    def get_download_targets(self, d_start: datetime) -> list[typing.Any]:
        return []

    def build_output_dir(self, d_start: datetime, ens_number: str) -> str:
        return str(self.out_dir)

    @abstractmethod
    def get_file_specs(self, d_start: datetime) -> list[FileSpec | tuple[str, str]]:
        """
        Return the files for one hour.

        Either FileSpec entries, or (subdir, filename) tuples where the same subdir is used remotely
        and under ``out_dir``.
        """
        pass

    def _specs(self, d_start: datetime) -> list[FileSpec]:
        specs: list[FileSpec] = []
        for spec in self.get_file_specs(d_start):
            if isinstance(spec, FileSpec):
                specs.append(spec)
            else:
                subdir, filename = spec
                specs.append(FileSpec(subdir, filename, subdir))
        return specs

    def _remote_url(self, spec: FileSpec) -> str:
        parts = [self.base_url.rstrip("/"), spec.remote_dir.strip("/"), spec.filename]
        return "/".join(p for p in parts if p)

    def _local_path(self, spec: FileSpec) -> str:
        return os.path.join(str(self.out_dir), spec.local_dir, spec.filename)

    def _cycle_files(self, d_start: datetime) -> list[tuple[str, str]]:
        return [(self._remote_url(spec), self._local_path(spec)) for spec in self._specs(d_start)]

    def _cleanup_old_data(self) -> None:
        """Remove files of hours older than the window; files the window needs are never removed."""
        if self.cleanback_hours <= self.lookback_hours:
            return
        keep = {
            os.path.normpath(path)
            for d in self._protected_cycles()
            if self.should_process_hour(d)
            for _, path in self._cycle_files(d)
        }
        for hour in range(self.cleanback_hours, self.lookback_hours, -1):
            d_start = self.start_time - timedelta(hours=hour)
            if not self.should_process_hour(d_start):
                continue
            for _, local_path in self._cycle_files(d_start):
                path = os.path.normpath(local_path)
                if path in keep or not os.path.isfile(path):
                    continue
                LOG.debug("Removing old file: %s", path)
                os.remove(path)
                parent = os.path.dirname(path)
                if parent != os.path.normpath(str(self.out_dir)) and not os.listdir(parent):
                    os.rmdir(parent)


class MRMSMultiSensorDownloader(FixedFileDownloader, ABC):
    """
    Hourly MRMS MultiSensor QPE (Pass1 and Pass2) for one MRMS domain.

    The server layout is ``<domain>/MultiSensor_QPE_01H_PassN_00.00/YYYYMMDD/<file>``. The local
    layout depends on which engine reader consumes the files and is selected by ``local_layout``.
    """

    #: MRMS domain directory on the server: CONUS, ALASKA, HAWAII or CARIB.
    region: str = ""

    #: Flat per-pass directories, read for SuppPcp 2, 5, 6 and 10.
    LAYOUT_FLAT: typing.ClassVar[str] = "MultiSensor_QPE_01H_{pass_}_00.00"
    #: Per-pass, per-date directories, read under MRMS_Hourly for SuppPcp 16 bias correction.
    LAYOUT_DATED: typing.ClassVar[str] = "MultiSensor_QPE_01H_{pass_}_00.00/{date}"
    #: Per-date directories without the product version, read for the MRMS half of SuppPcp 11 and 12.
    LAYOUT_STAGE4_BLEND: typing.ClassVar[str] = "MultiSensor_QPE_01H_{pass_}/{date}"

    local_layout: str = LAYOUT_FLAT

    @property
    def base_url(self) -> str:
        return f"https://noaa-mrms-pds.s3.amazonaws.com/{self.region}"

    def get_file_specs(self, d_start: datetime) -> list[FileSpec]:
        date = d_start.strftime("%Y%m%d")
        specs: list[FileSpec] = []
        for pass_ in ("Pass1", "Pass2"):
            product = f"MultiSensor_QPE_01H_{pass_}_00.00"
            filename = f"MRMS_{product}_{date}-{d_start.strftime('%H')}0000.grib2.gz"
            specs.append(FileSpec(f"{product}/{date}", filename, self.local_layout.format(pass_=pass_, date=date)))
        return specs


class NAMNestAnADownloader(ForecastDownloader, ABC):
    """NAM nest AnA: forecast hours 1-6 of the 00/06/12/18Z cycle covering each hour, for one nest."""

    retired_after = NAM_HIRESW_RETIRED_AFTER
    retirement_note = NAM_HIRESW_RETIREMENT_NOTE

    #: Nest name used in NAM file names, e.g. hawaiinest or priconest.
    domain: str = ""

    @property
    def base_url(self) -> str:
        return "https://noaa-nam-pds.s3.amazonaws.com"

    @staticmethod
    def nam_cycle(d_start: datetime) -> datetime:
        """Return the 00/06/12/18Z NAM cycle that covers d_start."""
        return d_start - timedelta(hours=d_start.hour % 6)

    def get_download_targets(self, d_start: datetime) -> list[typing.Any]:
        return list(range(1, 7))

    def build_output_dir(self, d_start: datetime, ens_number: str) -> str:
        return os.path.join(str(self.out_dir), f"nam.{self.nam_cycle(d_start).strftime('%Y%m%d')}")

    def build_file_url_and_name(self, d_start: datetime, target: typing.Any, ens_number: str) -> tuple[str, str]:
        cycle = self.nam_cycle(d_start)
        filename = f"nam.t{cycle.strftime('%H')}z.{self.domain}.hiresf{target:02d}.tm00.grib2"
        return f"{self.base_url}/nam.{cycle.strftime('%Y%m%d')}/{filename}", filename
