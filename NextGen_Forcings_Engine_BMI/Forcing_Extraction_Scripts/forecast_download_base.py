from __future__ import annotations

import argparse
import email.utils
import enum
import hashlib
import json
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
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

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


class Region(enum.Enum):
    """Geographic domain a downloader fetches a source for."""

    CONUS = "CONUS"
    ALASKA = "Alaska"
    HAWAII = "Hawaii"
    PUERTO_RICO = "Puerto_Rico"
    GUAM = "Guam"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def parse(cls, value: Region | str) -> Region:
        """Return the region named by ``value``, ignoring case and spaces/underscores/hyphens."""
        if isinstance(value, cls):
            return value
        key = re.sub(r"[\s_-]+", "", str(value)).lower()
        for region in cls:
            if key in (region.value.replace("_", "").lower(), region.name.replace("_", "").lower()):
                return region
        raise ValueError(f"Unknown region {value!r}; expected one of {', '.join(r.value for r in cls)}.")


class DownloadError(RuntimeError):
    """A required file could not be retrieved after all attempts."""


class ServerInconsistency(OSError):
    """A server response contradicts its own metadata (length, checksum, ranges, content type)."""


class _InconsistentNotModified(ServerInconsistency):
    """A 304 whose ETag contradicts the request; the file is then fetched unconditionally."""


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


def utc_now() -> datetime:
    """Current UTC time as a naive datetime, matching the cycle times used throughout."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


# ---------------------------------------------------------------------- HTTP validator metadata
#
# Each downloaded file gets a sidecar ``.<name>.meta.json`` next to it, recording the upstream
# validators (ETag, Last-Modified, Content-Length) from the response that produced it. The sidecar
# is written after the data file, so a crash leaves a file without metadata (treated as
# unvalidated), never metadata describing missing or partial data.

META_SUFFIX = ".meta.json"


def meta_path(out_path: str) -> str:
    return os.path.join(os.path.dirname(out_path), f".{os.path.basename(out_path)}{META_SUFFIX}")


def read_meta(out_path: str) -> dict[str, typing.Any] | None:
    try:
        with open(meta_path(out_path), encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def write_meta(out_path: str, meta: dict[str, typing.Any]) -> None:
    path = meta_path(out_path)
    temp = f"{path}.tmp.{uuid.uuid4()}"
    with open(temp, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1, sort_keys=True)
    os.replace(temp, path)


def conditional_headers(validators: dict[str, typing.Any] | None, no_cache: bool) -> dict[str, str]:
    """Request headers asking the server to answer 304 if the stored version is still current."""
    headers: dict[str, str] = {}
    if not validators:
        return headers
    if validators.get("etag"):
        headers["If-None-Match"] = validators["etag"]
    if validators.get("last_modified"):
        headers["If-Modified-Since"] = validators["last_modified"]
    if headers and no_cache:
        headers["Cache-Control"] = "no-cache"
    return headers


_MULTIPART_ETAG = re.compile(r'^"?([0-9a-f]{32})-(\d+)"?$')
#: Part sizes (MiB) commonly used by S3 upload tools, tried when inferring a multipart ETag.
_COMMON_PART_SIZES_MIB = (5, 8, 16, 32, 64, 100, 128, 256, 512)


def multipart_part_sizes(size: int, parts: int) -> list[int]:
    """Part sizes in bytes that split ``size`` bytes into exactly ``parts`` parts, likeliest first."""
    if parts < 1 or size < parts:
        return []
    mib = 1 << 20
    smallest = -(-size // parts)
    candidates = [smallest, -(-smallest // mib) * mib] + [m * mib for m in _COMMON_PART_SIZES_MIB]
    sizes: list[int] = []
    for part_size in candidates:
        if -(-size // part_size) == parts and part_size not in sizes:
            sizes.append(part_size)
    return sizes


def s3_multipart_etag(fh: typing.BinaryIO, part_size: int) -> str:
    """The ETag S3 assigns to a multipart upload of ``fh``'s content with ``part_size``-byte parts."""
    fh.seek(0)
    digests = b""
    parts = 0
    while True:
        part = hashlib.md5()
        remaining = part_size
        while remaining:
            chunk = fh.read(min(CHUNK_SIZE, remaining))
            if not chunk:
                break
            part.update(chunk)
            remaining -= len(chunk)
        if remaining == part_size:
            break
        digests += part.digest()
        parts += 1
        if remaining:
            break
    return f"{hashlib.md5(digests).hexdigest()}-{parts}"


def etag_matches(fh: typing.BinaryIO, etag: str | None) -> bool | None:
    """
    Whether ``fh``'s content matches an S3-style ETag.

    For a plain-MD5 ETag: True or False. For a multipart ETag ("<md5>-<parts>"): True when one of
    the likely part sizes reproduces it, else None, since the upload part size is unknown. None for
    any other ETag.
    """
    expected = plain_md5_etag(etag)
    if expected is not None:
        fh.seek(0)
        digest = hashlib.md5()
        for chunk in iter(lambda: fh.read(CHUNK_SIZE), b""):
            digest.update(chunk)
        return digest.hexdigest() == expected
    match = _MULTIPART_ETAG.match(etag or "")
    if not match:
        return None
    size = fh.seek(0, os.SEEK_END)
    wanted = f"{match.group(1)}-{match.group(2)}"
    for part_size in multipart_part_sizes(size, int(match.group(2))):
        if s3_multipart_etag(fh, part_size) == wanted:
            return True
    return None


def set_mtime_from_header(path: str, last_modified: str | None) -> None:
    if not last_modified:
        return
    try:
        stamp = email.utils.parsedate_to_datetime(last_modified).timestamp()
    except (TypeError, ValueError):
        return
    os.utime(path, (stamp, stamp))


@dataclass
class FetchResult:
    """Outcome of one successful HTTP retrieval: 200 (body written) or 304 (unchanged)."""

    status: int
    headers: typing.Mapping[str, str]
    md5_verified: bool = False

    def validators(self, url: str, size: int | None) -> dict[str, typing.Any]:
        now = utc_now().isoformat(timespec="seconds")
        return {
            "url": url,
            "etag": self.headers.get("ETag"),
            "last_modified": self.headers.get("Last-Modified"),
            "content_length": size,
            "md5_verified": self.md5_verified,
            "fetched_at": now,
            "checked_at": now,
        }


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
    """
    Abstract base class for structured forecast data downloads.

    A concrete subclass describes one source (model or observation product). ``REGIONS`` maps each
    region the source is fetched for to that region's settings, which an instance reads from
    ``self.domain``. A source without regional variants (a global model) has the single key
    ``None``. The region is a required argument: a Region, or None for a source without variants.
    """

    #: Region -> settings of this source for that region; ``{None: ...}`` for a global source.
    REGIONS: typing.ClassVar[typing.Mapping[Region | None, typing.Any]] = {}

    default_lookback: int = 24
    default_cleanback: int = 240
    default_lagback: int = 6

    #: Why the product ended, with a reference; included in the error for later cycles.
    retirement_note: typing.ClassVar[str] = ""

    #: True for products whose files are rewritten in place upstream (Stage IV reruns, NDFD).
    #: Existing files of such products are revalidated with conditional requests on every run.
    mutable_upstream: bool = False
    #: A 404 for a cycle older than this many hours is final and is not retried.
    archived_after_hours: int = 48

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
        revalidate: bool = False,
        *,
        region: Region | str | None,
    ) -> None:
        #: Region this instance downloads (None for a global source), and the source's settings for it.
        self.region: Region | None = self.resolve_region(region)
        self.domain: typing.Any = self.REGIONS[self.region]
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
        #: Revalidate existing files of immutable products too (normally skipped without a request).
        self.revalidate: bool = revalidate

        self.last_hour: bool | None = None
        self._cycle_time: datetime | None = None

    @classmethod
    def resolve_region(cls, region: Region | str | None) -> Region | None:
        """
        Return ``region`` as a Region, rejecting regions the source does not cover.

        A source without regional variants accepts only None, and other sources do not accept None.
        """
        if None in cls.REGIONS:
            if region is not None:
                raise ValueError(f"{cls.__name__} has no regional variants; its region must be None, not {region}.")
            return None
        covered = ", ".join(str(r) for r in cls.REGIONS)
        if region is None:
            raise ValueError(f"{cls.__name__} needs a region: one of {covered}.")
        resolved = Region.parse(region)
        if resolved not in cls.REGIONS:
            raise ValueError(f"{cls.__name__} does not cover region {resolved}; it covers {covered}.")
        return resolved

    @classmethod
    def label(cls, region: Region | str | None) -> str:
        """A short name for this source in one region, e.g. ``HRRR[Alaska]``, or ``GFS`` for a global source."""
        name = cls.__name__.removesuffix("Downloader")
        resolved = cls.resolve_region(region)
        return name if resolved is None else f"{name}[{resolved}]"

    @classmethod
    def retirement_date(cls, region: Region | None) -> datetime | None:
        """
        Last cycle the source has data for in ``region``, or None if it is still produced there.

        A retired source overrides this; cycles after the date fail at once, with ``retirement_note``.
        """
        return None

    @classmethod
    def window_defaults(cls, region: Region | None) -> tuple[int, int, int]:
        """Standalone defaults (lookback, cleanback, lagback) in hours for ``region``."""
        return cls.default_lookback, cls.default_cleanback, cls.default_lagback

    def effective_lagback(self) -> int:
        """Return the lag-back (in hours) that ends the processing window; subclasses may override."""
        return self.lagback_hours

    @classmethod
    def from_cli_args(
        cls,
        argv: typing.Sequence[str] | None = None,
        prog: str | None = None,
    ) -> typing.Self:
        """Create an instance of the subclass using command-line arguments."""
        parser = argparse.ArgumentParser(prog=prog, description=cls.__doc__)
        parser.add_argument("outDir", type=str, help="Output directory path")
        parser.add_argument("startTime", type=lambda s: datetime.strptime(s, "%Y-%m-%d %H:%M:%S"))
        if len(cls.REGIONS) == 1:  # nothing to choose: a global source, or one that covers a single region
            parser.set_defaults(region=next(iter(cls.REGIONS)))
        else:
            parser.add_argument("--region", type=cls.resolve_region, required=True,
                                help=f"One of: {', '.join(str(r) for r in cls.REGIONS)}.")
        parser.add_argument("--lookBackHours", type=int, default=None, help="Default depends on source and region.")
        parser.add_argument("--cleanBackHours", type=int, default=None, help="Default depends on source and region.")
        parser.add_argument("--lagBackHours", type=int, default=None, help="Default depends on source and region.")
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
        parser.add_argument("--revalidate", action="store_true", default=False,
                            help="Check existing files against upstream with conditional requests.")
        args = parser.parse_args(argv)
        lookback, cleanback, lagback = cls.window_defaults(args.region)
        args.lookBackHours = lookback if args.lookBackHours is None else args.lookBackHours
        args.cleanBackHours = cleanback if args.cleanBackHours is None else args.cleanBackHours
        args.lagBackHours = lagback if args.lagBackHours is None else args.lagBackHours
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
            revalidate=args.revalidate,
            region=args.region,
        )

    @classmethod
    def cli_main(cls, argv: typing.Sequence[str] | None = None, prog: str | None = None) -> None:
        """Command-line entry point: parse arguments, run, and exit non-zero on failure."""
        configure_logging()
        try:
            ok = cls.from_cli_args(argv, prog).run()
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
                self._cycle_time = d_start
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
        last_cycle = self.retirement_date(self.region)
        if last_cycle is None or d_start <= last_cycle:
            return None
        return (f"{self.label(self.region)}: no data after the {last_cycle:%Y-%m-%d %HZ} cycle "
                f"({self.retirement_note}); cannot supply {d_start:%Y-%m-%d %HZ}")

    def _warn_nothing_processed(self) -> None:
        first = self.start_time - timedelta(hours=self.lookback_hours)
        last = self.start_time - timedelta(hours=self.effective_lagback() + 1)
        LOG.warning(
            "%s: no cycle between %s and %s matches this product's cycle schedule; nothing was downloaded.",
            self.label(self.region), first, last,
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
            self.label(self.region), self.lookback_hours, self.effective_lagback(),
        )
        processed = 0
        for d_start, files in self._cycles():
            processed += 1
            problem = self.retirement_problem(d_start)
            if problem:
                raise DownloadError(problem)
            self.pre_download_hook(d_start)
            for url, out_path in files:
                if self._needs_download(out_path, url):
                    self._download_file(url, out_path)
            self.post_download_hook(d_start)

        if processed == 0:
            self._warn_nothing_processed()

    def _cycle_is_archived(self) -> bool:
        """True when the cycle being fetched is old enough that a 404 means the file will never exist."""
        if self._cycle_time is None:
            return False
        return utc_now() - self._cycle_time > timedelta(hours=self.archived_after_hours)

    @staticmethod
    def _stored_validators(out_path: str, url: str) -> dict[str, typing.Any] | None:
        """Validators recorded for ``url`` in the sidecar of ``out_path``, if any."""
        meta = read_meta(out_path)
        if not meta:
            return None
        if "parts" in meta:
            return meta["parts"].get(url)
        return meta if meta.get("url") == url else None

    def _needs_download(self, out_path: str, url: str) -> bool:
        """
        Decide whether ``url`` must be requested for ``out_path``.

        Missing files are always requested. Existing files of mutable products (or with
        ``revalidate``) are requested conditionally. Existing files of immutable products are only
        requested again when their size disagrees with the recorded upstream size.
        """
        if not os.path.isfile(out_path):
            return True
        if self.mutable_upstream or self.revalidate:
            return True
        stored = self._stored_validators(out_path, url)
        size = stored.get("content_length") if stored and "parts" not in (read_meta(out_path) or {}) else None
        if size is not None and os.path.getsize(out_path) != size:
            LOG.warning("%s is %d bytes but upstream reported %d; downloading it again.",
                        out_path, os.path.getsize(out_path), size)
            return True
        LOG.debug("Skipping existing: %s", out_path)
        return False

    @staticmethod
    def _verify_md5(path: str, etag: str | None) -> bool:
        """
        Check the file against an S3-style ETag (see etag_matches); return True if verified.

        A plain-MD5 ETag that does not match the content raises ServerInconsistency. A multipart
        ETag that no likely part size reproduces leaves the content unverified, not rejected.
        """
        with open(path, "rb") as fh:
            verdict = etag_matches(fh, etag)
        if verdict is False:
            raise ServerInconsistency(f"MD5 mismatch: content does not match ETag {etag}")
        if verdict is None and _MULTIPART_ETAG.match(etag or ""):
            LOG.debug("Multipart ETag %s not reproduced for %s with likely part sizes; content unverified.",
                      etag, path)
        return bool(verdict)

    def _fetch(self, url: str, dest_path: str,
               conditional: typing.Mapping[str, str] | None = None) -> FetchResult:
        """
        Download ``url`` to ``dest_path`` with retries.

        ``conditional`` request headers may yield a 304, returned without writing ``dest_path``.
        A body interrupted mid-transfer is resumed with ``Range`` + ``If-Range`` when the server
        offers byte ranges and an ETag. Completed bodies are checked against Content-Length, their
        file type and, when the ETag is a plain MD5, their content. A 404 for an archived cycle is
        final. Raises DownloadError after the last failed attempt.
        """
        attempts: int = self.max_download_attempts
        last_error = "unknown error"
        resume_etag: str | None = None
        for attempt in range(1, attempts + 1):
            headers = {**REQUEST_HEADERS, **(conditional or {})}
            offset = os.path.getsize(dest_path) if resume_etag and os.path.exists(dest_path) else 0
            if offset:
                headers["Range"] = f"bytes={offset}-"
                headers["If-Range"] = resume_etag
            resp_etag: str | None = None
            ranges_ok = False
            body_started = False
            retry_now = False
            try:
                LOG.info("%s %s (attempt %d/%d%s)", "Revalidating" if conditional else "Downloading",
                         url, attempt, attempts, f", resuming at byte {offset}" if offset else "")
                with requests.get(url, stream=True, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
                                  headers=headers) as resp:
                    if resp.status_code == 304:
                        if not conditional:
                            raise ServerInconsistency("answered 304 Not Modified to a request without conditions")
                        asked, answered = conditional.get("If-None-Match"), resp.headers.get("ETag")
                        if asked and answered and answered != asked:
                            raise _InconsistentNotModified(
                                f"answered 304 Not Modified with ETag {answered} to If-None-Match {asked}; "
                                "fetching it unconditionally")
                        return FetchResult(304, resp.headers)
                    resp.raise_for_status()
                    if resp.status_code == 206:
                        self._check_content_range(resp.headers, offset)
                    resp_etag = resp.headers.get("ETag")
                    ranges_ok = (resp.headers.get("Accept-Ranges", "").lower() == "bytes"
                                 or resp.status_code == 206)
                    if resp.status_code != 206:
                        offset = 0
                    expected = resp.headers.get("Content-Length")
                    total = offset + int(expected) if expected is not None else None
                    written = offset
                    with open(dest_path, "ab" if offset else "wb") as fh:
                        body_started = True
                        while True:
                            chunk = resp.raw.read(CHUNK_SIZE)
                            if not chunk:
                                break
                            fh.write(chunk)
                            written += len(chunk)
                    body_started = False
                    if total is not None and written != total:
                        received = f"{written - offset} bytes from byte {offset}" if offset else f"{written} bytes"
                        raise ServerInconsistency(f"received {received} but Content-Length is {expected}")
                    with open(dest_path, "rb") as fh:
                        problem = content_signature_problem(url, fh.read(SNIFF_BYTES))
                    if problem:
                        raise ServerInconsistency(f"content does not match {url.rsplit('/', 1)[-1]}: {problem}")
                    md5_verified = self._verify_md5(dest_path, resp_etag)
                    return FetchResult(200, resp.headers, md5_verified)
            except requests.HTTPError as exc:
                last_error = describe_request_error(exc)
                if exc.response is not None and exc.response.status_code in (404, 410) and self._cycle_is_archived():
                    raise DownloadError(
                        f"Failed to download {url}: {last_error}; the cycle is older than "
                        f"{self.archived_after_hours} h, so it was not retried."
                    ) from exc
            except (requests.RequestException, urllib3.exceptions.HTTPError, OSError) as exc:
                # Errors while streaming the body come from urllib3, not requests.
                last_error = describe_request_error(exc)
                if isinstance(exc, ServerInconsistency):
                    LOG.error("%s: %s", url, last_error)
                if isinstance(exc, _InconsistentNotModified):
                    conditional = None
                    retry_now = True
            keep_partial = body_started and ranges_ok and bool(resp_etag)
            resume_etag = resp_etag if keep_partial else None
            if not keep_partial and os.path.exists(dest_path):
                os.remove(dest_path)
            LOG.warning("Attempt %d/%d for %s failed: %s", attempt, attempts, url, last_error)
            if attempt < attempts and not retry_now:
                time.sleep(self.download_attempt_interval)
        raise DownloadError(f"Failed to download {url} after {attempts} attempt(s): {last_error}")

    @staticmethod
    def _check_content_range(headers: typing.Mapping[str, str], offset: int) -> None:
        """Verify a 206 response continues the file from ``offset`` to the end of the object."""
        if not offset:
            raise ServerInconsistency("answered 206 Partial Content to a request without Range")
        content_range = headers.get("Content-Range", "")
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+|\*)", content_range.strip())
        if not match or int(match.group(1)) != offset:
            raise ServerInconsistency(f"asked for bytes {offset}- but got Content-Range {content_range!r}")
        last, complete = int(match.group(2)), match.group(3)
        if complete != "*" and last != int(complete) - 1:
            raise ServerInconsistency(f"Content-Range {content_range!r} stops before the end of the object")
        length = headers.get("Content-Length")
        if length is not None and int(length) != last - offset + 1:
            raise ServerInconsistency(f"Content-Range {content_range!r} disagrees with Content-Length {length}")

    @staticmethod
    def _publish(temp_path: str, out_path: str, replace: bool = False) -> None:
        """
        Move a completed temporary file into place.

        A new file is hard-linked so a concurrent writer is never clobbered; ``replace`` (an
        upstream update of an existing file) renames over it atomically.
        """
        if replace:
            os.replace(temp_path, out_path)
            return
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
        Download a file (or all its parts) to temporary paths, publish it atomically and record its
        validators.

        An existing file is requested conditionally and left untouched when every part answers 304,
        or kept when it cannot be revalidated. A multi-part file is rebuilt from all its parts when
        any changed, and published only when every part downloaded. Returns True when the file is
        in place, False when a tolerated sub-hourly file is missing. Raises DownloadError otherwise.
        """
        out_dir: str = os.path.dirname(out_path)
        os.makedirs(out_dir, exist_ok=True)
        parts = self._part_urls(url, out_path)
        existing = os.path.isfile(out_path)
        stored = {u: self._stored_validators(out_path, u) for u in parts} if existing else {}
        single = stored.get(url) if len(parts) == 1 else None
        if single and single.get("content_length") is not None and os.path.getsize(out_path) != single["content_length"]:
            stored = {}  # local copy is damaged; fetch unconditionally
        name, token = os.path.basename(out_path), uuid.uuid4()
        temp_path: str = os.path.join(out_dir, f".{name}.tmp.{token}")
        part_paths = [temp_path] if len(parts) == 1 else [
            os.path.join(out_dir, f".{name}.part{i}.{token}") for i in range(len(parts))]
        try:
            results = {
                u: self._fetch(u, p, conditional_headers(stored.get(u), self.mutable_upstream))
                for u, p in zip(parts, part_paths)
            }
            if existing and all(r.status == 304 for r in results.values()):
                meta = read_meta(out_path) or {}
                meta["checked_at"] = utc_now().isoformat(timespec="seconds")
                write_meta(out_path, meta)
                LOG.debug("Unchanged upstream: %s", out_path)
                return True
            for u, p in zip(parts, part_paths):
                if results[u].status == 304:  # every part's bytes are needed to rebuild the file
                    results[u] = self._fetch(u, p)
                self._check_part(u, results[u].headers)
            sizes = {u: os.path.getsize(p) for u, p in zip(parts, part_paths)}
            if len(parts) > 1:
                with open(temp_path, "wb") as outfile:
                    for part_path in part_paths:
                        with open(part_path, "rb") as infile:
                            shutil.copyfileobj(infile, outfile)
            self._publish(temp_path, out_path, replace=existing)
            if len(parts) == 1:
                set_mtime_from_header(out_path, results[url].headers.get("Last-Modified"))
                write_meta(out_path, results[url].validators(url, sizes[url]))
            else:
                write_meta(out_path, {"url": url, "parts": {u: r.validators(u, sizes[u]) for u, r in results.items()}})
        except DownloadError as exc:
            if existing:
                LOG.warning("Could not revalidate %s (%s); keeping the existing file.", out_path, exc)
                return True
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
        LOG.info("%s %s", "Updated" if existing else "Downloaded", out_path)
        return True

    # ------------------------------------------------------------------ availability check

    MISSING_UPSTREAM: typing.ClassVar[str] = "missing upstream"
    NOT_DOWNLOADED: typing.ClassVar[str] = "available, not downloaded"
    CURRENT: typing.ClassVar[str] = "downloaded and current"
    UNVALIDATED: typing.ClassVar[str] = "downloaded, no stored validators"
    STALE: typing.ClassVar[str] = "downloaded but changed upstream"
    SIZE_MISMATCH: typing.ClassVar[str] = "downloaded but size differs"

    def _check_data_availability(self) -> bool:
        """
        Compare every file of the window with upstream using HEAD requests, without downloading.

        Each remote part is classified as missing upstream, available but not downloaded,
        downloaded and current, downloaded without stored validators, changed upstream, or
        differing in size. Returns True if nothing is missing upstream.
        """
        LOG.info(
            "%s: checking data availability. lookback: %s lagback: %s",
            self.label(self.region), self.lookback_hours, self.effective_lagback(),
        )
        counts: Counter[str] = Counter()
        for d_start, files in self._cycles():
            problem = self.retirement_problem(d_start)
            if problem:
                LOG.error("%s", problem)
                counts[self.MISSING_UPSTREAM] += sum(len(self._part_urls(u, p)) for u, p in files)
                continue
            for url, out_path in files:
                for part in self._part_urls(url, out_path):
                    status = self._classify(part, out_path, self._head(part))
                    counts[status] += 1
                    if status in (self.STALE, self.SIZE_MISMATCH):
                        LOG.warning("%s: %s (%s)", status, out_path, part)
                    elif status == self.NOT_DOWNLOADED:
                        LOG.debug("%s: %s", status, part)

        total = sum(counts.values())
        LOG.info("--- Forcing Availability Verification Summary ---")
        LOG.info("Total files checked: %d", total)
        for status in (self.MISSING_UPSTREAM, self.NOT_DOWNLOADED, self.CURRENT, self.UNVALIDATED,
                       self.STALE, self.SIZE_MISMATCH):
            if counts[status]:
                LOG.info("  %s: %d", status, counts[status])
        missing = counts[self.MISSING_UPSTREAM]
        if missing:
            LOG.warning("Verification complete: %d file(s) are currently missing from the server.", missing)
            return False
        LOG.info("Verification complete: all %d requested files are available upstream.", total)
        return True

    def _classify(self, url: str, out_path: str, upstream: typing.Mapping[str, str] | None) -> str:
        if upstream is None:
            return self.MISSING_UPSTREAM
        if not os.path.isfile(out_path):
            return self.NOT_DOWNLOADED
        stored = self._stored_validators(out_path, url)
        if not stored:
            return self.UNVALIDATED
        is_part = "parts" in (read_meta(out_path) or {})
        upstream_size = upstream.get("Content-Length")
        if not is_part and upstream_size is not None and os.path.getsize(out_path) != int(upstream_size):
            return self.SIZE_MISMATCH
        for key, header in (("etag", "ETag"), ("last_modified", "Last-Modified")):
            if stored.get(key) and upstream.get(header) and stored[key] != upstream[header]:
                return self.STALE
        return self.CURRENT

    def _head(self, url: str) -> typing.Mapping[str, str] | None:
        """HEAD ``url`` with retries; return its headers, or None if it is not available."""
        attempts: int = self.max_download_attempts
        last_error = "unknown error"
        for attempt in range(1, attempts + 1):
            try:
                response: requests.Response = requests.head(url, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
                                                             headers=REQUEST_HEADERS)
                if response.status_code == requests.codes.ok:
                    LOG.debug("File is available: %s", url)
                    return response.headers
                last_error = f"HTTP {response.status_code}"
            except requests.RequestException as exc:
                last_error = describe_request_error(exc)
            LOG.debug("Availability attempt %d/%d for %s failed: %s", attempt, attempts, url, last_error)
            if attempt < attempts:
                time.sleep(self.download_attempt_interval)

        level = logging.WARNING if (self.last_hour and self.subhourly_avg) else logging.ERROR
        LOG.log(level, "Not available after %d attempt(s): %s (%s)", attempts, url, last_error)
        return None


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
                if os.path.exists(meta_path(path)):
                    os.remove(meta_path(path))
                parent = os.path.dirname(path)
                if parent != os.path.normpath(str(self.out_dir)) and not os.listdir(parent):
                    os.rmdir(parent)
