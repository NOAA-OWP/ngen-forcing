"""
Probe every downloader, in every region it covers, against its live upstream server without
downloading whole files.

For each source and region, a recent cycle that the product should already have published is chosen,
the files the downloader would fetch for that cycle are enumerated with the downloader's own URL
logic, and a sample of them (or all, with ``--all``) is checked:

1. ``HEAD`` must return 200 (the file exists; its size is reported).
2. A ranged ``GET`` of the first bytes must look like GRIB edition 2 (after gunzip for ``.gz`` files).

Run from ``NextGen_Forcings_Engine_BMI/``::

    python -m Forcing_Extraction_Scripts.probe_sources                 # every source and region
    python -m Forcing_Extraction_Scripts.probe_sources HRRR NBM[Alaska] --all
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests
import urllib3

from Forcing_Extraction_Scripts.forecast_download_base import (
    CONNECT_TIMEOUT,
    READ_TIMEOUT,
    REQUEST_HEADERS,
    SNIFF_BYTES,
    ForecastDownloader,
    content_signature_problem,
)
from Forcing_Extraction_Scripts.sources import Source, all_sources, parse_source

#: How old (hours) a cycle must be before it is expected to be complete upstream.
DEFAULT_AGE_HOURS = 26
#: Products whose upstream only holds the latest issuance are probed at the newest cycle.
AGE_HOURS_OVERRIDE: dict[str, int] = {"NDFDDownloader": 0}
#: How far back to search for a cycle the product runs at.
SEARCH_HOURS = 48


TIMEOUT = (CONNECT_TIMEOUT, READ_TIMEOUT)

#: Hosts with request-rate limits: (minimum seconds between requests, files sampled per source).
RATE_LIMITED_HOSTS: dict[str, tuple[float, int]] = {"nomads.ncep.noaa.gov": (5.0, 1)}


class PoliteSession(requests.Session):
    """A requests session that spaces out requests to rate-limited hosts."""

    def __init__(self) -> None:
        super().__init__()
        self._last_request: dict[str, float] = {}

    def request(self, method, url, *args, **kwargs):
        host = urlparse(url).netloc
        if host in RATE_LIMITED_HOSTS:
            wait = self._last_request.get(host, 0.0) + RATE_LIMITED_HOSTS[host][0] - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last_request[host] = time.monotonic()
        return super().request(method, url, *args, **kwargs)


def head(session: requests.Session, url: str) -> requests.Response:
    """HEAD ``url`` as the downloaders would (identity encoding, timeouts)."""
    return session.head(url, headers=REQUEST_HEADERS, timeout=TIMEOUT, allow_redirects=True)


def get_prefix(session: requests.Session, url: str, extra_headers: dict[str, str] | None = None,
               read: int = 0) -> tuple[int, dict[str, str], bytes]:
    """GET ``url`` and return (status, headers, up to ``read`` bytes of a 200/206 body)."""
    with session.get(url, headers={**REQUEST_HEADERS, **(extra_headers or {})}, timeout=TIMEOUT,
                     stream=True) as resp:
        body = resp.raw.read(read) if read and resp.status_code in (200, 206) else b""
        return resp.status_code, dict(resp.headers), body


@dataclass
class SourceCycle:
    """The cycle chosen for a source and the remote files its downloader would fetch for it."""

    downloader: str
    cycle: datetime
    urls: list[str]
    note: str = ""


@dataclass
class FileProbe:
    url: str
    ok: bool
    status: int | None = None
    size: int | None = None
    detail: str = ""


@dataclass
class SourceProbe:
    source: str
    downloader: str = ""
    registered_as: list[str] = field(default_factory=list)
    cycle: str = ""
    files_for_cycle: int = 0
    probes: list[FileProbe] = field(default_factory=list)
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and bool(self.probes) and all(p.ok for p in self.probes)


def registrations() -> dict[Source, list[str]]:
    """Map (downloader class, region) -> forcing identifiers it is registered under in forcing_extraction."""
    import forcing_extraction as fe

    regs: dict[Source, list[str]] = {}
    for label, table in (("forecast", fe.FORCING_SRC), ("AnA", fe.FORCING_ANA_SRC),
                         ("Stage IV MRMS", fe.STAGE4_MRMS_SRC)):
        for key, source in table.items():
            regs.setdefault(source, []).append(f"{label} {key}")
    return regs


def make_probe_downloader(source: Source, cycle: datetime, out_dir: str) -> ForecastDownloader:
    """Instantiate a downloader whose processing window is exactly ``cycle``."""
    cls, region = source
    return cls(out_dir=out_dir, start_time=cycle + timedelta(hours=1), lookback_hours=1, ens_number=1,
               max_download_attempts=1, download_attempt_interval=0, check_file_availability=1, region=region)


def pick_cycle(source: Source, now: datetime, age_hours: int, out_dir: str) -> datetime | None:
    """
    Most recent cycle at least ``age_hours`` old that the downloader would fetch files for.

    For a retired product, the search starts at its last cycle instead, so the probe exercises the
    historical data retrospective runs use.
    """
    cls, region = source
    start = (now - timedelta(hours=age_hours)).replace(minute=0, second=0, microsecond=0, tzinfo=None)
    last_cycle = cls.retirement_date(region)
    if last_cycle is not None:
        start = min(start, last_cycle)
    for back in range(SEARCH_HOURS):
        cycle = start - timedelta(hours=back)
        if next(make_probe_downloader(source, cycle, out_dir).iter_remote_files(), None) is not None:
            return cycle
    return None


def probe_url(url: str, session: requests.Session) -> FileProbe:
    try:
        response = head(session, url)
    except requests.RequestException as exc:
        return FileProbe(url, False, detail=f"HEAD failed: {exc}")
    size = int(response.headers["Content-Length"]) if "Content-Length" in response.headers else None
    if response.status_code != 200:
        return FileProbe(url, False, response.status_code, size, f"HEAD returned {response.status_code}")
    try:
        _, _, data = get_prefix(session, url, {"Range": f"bytes=0-{SNIFF_BYTES - 1}"}, read=SNIFF_BYTES)
    except (requests.RequestException, urllib3.exceptions.HTTPError) as exc:
        return FileProbe(url, False, response.status_code, size, f"ranged GET failed: {exc}")
    problem = content_signature_problem(url, data, require_edition2=True)
    return FileProbe(url, not problem, response.status_code, size, problem or "GRIB2")


def sample(urls: list[str]) -> list[str]:
    """First, middle and last entries, without duplicates; fewer for rate-limited hosts."""
    picks = list(dict.fromkeys([urls[0], urls[len(urls) // 2], urls[-1]])) if urls else []
    limit = RATE_LIMITED_HOSTS.get(urlparse(urls[0]).netloc, (0.0, len(picks)))[1] if urls else 0
    return picks[:limit]


def source_cycle(source: Source, now: datetime | None = None) -> SourceCycle:
    """Choose a recently published cycle for ``source`` and list the files its downloader would fetch."""
    cls, region = source
    with tempfile.TemporaryDirectory() as out_dir:
        age = AGE_HOURS_OVERRIDE.get(cls.__name__, DEFAULT_AGE_HOURS)
        cycle = pick_cycle(source, now or datetime.now(timezone.utc), age, out_dir)
        if cycle is None:
            raise LookupError(f"no cycle in the {SEARCH_HOURS} h before now-{age} h yields any files")
        urls = list(dict.fromkeys(u for _, u, _ in make_probe_downloader(source, cycle, out_dir).iter_remote_files()))
    last_cycle = cls.retirement_date(region)
    note = f"retired after {last_cycle:%Y-%m-%d %HZ}" if last_cycle is not None else ""
    return SourceCycle(cls.__name__, cycle, urls, note)


def source_label(source: Source) -> str:
    """``Name[Region]`` for a (downloader class, region) pair."""
    return source[0].label(source[1])


def probe_source(source: Source, *, probe_all: bool = False, now: datetime | None = None,
                 regs: dict[Source, list[str]] | None = None) -> SourceProbe:
    result = SourceProbe(source_label(source), registered_as=(regs or {}).get(source, []))
    try:
        chosen = source_cycle(source, now)
    except Exception as exc:  # report and continue with the other sources
        result.error = f"{type(exc).__name__}: {exc}"
        return result
    result.downloader = chosen.downloader
    result.cycle = chosen.cycle.strftime("%Y-%m-%d %HZ") + (f" ({chosen.note})" if chosen.note else "")
    result.files_for_cycle = len(chosen.urls)
    with PoliteSession() as session:
        result.probes = [probe_url(u, session) for u in (chosen.urls if probe_all else sample(chosen.urls))]
    return result


def sources_from_args(specs: list[str]) -> list[Source]:
    """Sources named on a command line (all of them when none are named)."""
    if not specs:
        return all_sources()
    return list(dict.fromkeys(source for spec in specs for source in parse_source(spec)))


def format_report(results: list[SourceProbe]) -> str:
    lines = []
    for r in results:
        status = "OK  " if r.ok else "FAIL"
        reg = ", ".join(r.registered_as) or "not registered"
        lines.append(f"{status} {r.source} ({reg})")
        if r.error:
            lines.append(f"       {r.error}")
            continue
        good = sum(p.ok for p in r.probes)
        lines.append(f"       cycle {r.cycle}: {r.files_for_cycle} file(s); {good}/{len(r.probes)} probed OK")
        for p in r.probes:
            if not p.ok:
                lines.append(f"       - {p.detail}: {p.url}")
    failed = sum(not r.ok for r in results)
    lines.append(f"\n{len(results) - failed}/{len(results)} source(s) passed.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("sources", nargs="*",
                        help="Sources as NAME (every region) or NAME[Region], e.g. HRRR or NBM[Alaska] (default: all).")
    parser.add_argument("--all", action="store_true", help="Probe every file of the cycle, not a sample of three.")
    parser.add_argument("--json", type=Path, help="Also write results as JSON to this path.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    regs = registrations()
    try:
        sources = sources_from_args(args.sources)
    except ValueError as exc:
        parser.error(str(exc))
    results = [probe_source(s, probe_all=args.all, regs=regs) for s in sources]
    print(format_report(results))
    if args.json:
        args.json.write_text(json.dumps(
            [{**asdict(r), "ok": r.ok} for r in results], indent=2))
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
