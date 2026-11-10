"""
Probe every downloader script against its live upstream server without downloading whole files.

For each script, a recent cycle that the product should already have published is chosen, the files
the downloader would fetch for that cycle are enumerated with the downloader's own URL logic, and a
sample of them (or all, with ``--all``) is checked:

1. ``HEAD`` must return 200 (the file exists; its size is reported).
2. A ranged ``GET`` of the first bytes must look like GRIB edition 2 (after gunzip for ``.gz`` files).

Run from ``NextGen_Forcings_Engine_BMI/``::

    python -m Forcing_Extraction_Scripts.probe_sources                 # every script
    python -m Forcing_Extraction_Scripts.probe_sources CONUS/get_conus_HRRR.py --all
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

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

SCRIPT_DIR = Path(__file__).resolve().parent

#: How old (hours) a cycle must be before it is expected to be complete upstream.
DEFAULT_AGE_HOURS = 26
#: Products whose upstream only holds the latest issuance are probed at the newest cycle.
AGE_HOURS_OVERRIDE: dict[str, int] = {"NDFDConusDownloader": 0}
#: How far back to search for a cycle the product runs at.
SEARCH_HOURS = 48


@dataclass
class FileProbe:
    url: str
    ok: bool
    status: int | None = None
    size: int | None = None
    detail: str = ""


@dataclass
class ScriptProbe:
    script: str
    downloader: str = ""
    registered_as: list[str] = field(default_factory=list)
    cycle: str = ""
    files_for_cycle: int = 0
    probes: list[FileProbe] = field(default_factory=list)
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and bool(self.probes) and all(p.ok for p in self.probes)


def all_scripts() -> list[str]:
    """Every downloader script under Forcing_Extraction_Scripts, as paths relative to it."""
    return sorted(str(p.relative_to(SCRIPT_DIR)) for p in SCRIPT_DIR.glob("*/get_*.py"))


def registrations() -> dict[str, list[str]]:
    """Map script -> forcing identifiers it is registered under in forcing_extraction."""
    import forcing_extraction as fe

    regs: dict[str, list[str]] = {}
    for label, table in (("forecast", fe.FORCING_SRC), ("AnA", fe.FORCING_ANA_SRC),
                         ("Stage IV MRMS", fe.STAGE4_MRMS_SRC)):
        for key, script in table.items():
            regs.setdefault(script, []).append(f"{label} {key}")
    return regs


def make_probe_downloader(cls: type[ForecastDownloader], cycle: datetime, out_dir: str) -> ForecastDownloader:
    """Instantiate a downloader whose processing window is exactly ``cycle``."""
    return cls(out_dir=out_dir, start_time=cycle + timedelta(hours=1), lookback_hours=1, ens_number=1,
               max_download_attempts=1, download_attempt_interval=0, check_file_availability=1)


def pick_cycle(cls: type[ForecastDownloader], now: datetime, age_hours: int, out_dir: str) -> datetime | None:
    """Most recent cycle at least ``age_hours`` old that the downloader would fetch files for."""
    start = (now - timedelta(hours=age_hours)).replace(minute=0, second=0, microsecond=0, tzinfo=None)
    for back in range(SEARCH_HOURS):
        cycle = start - timedelta(hours=back)
        if next(make_probe_downloader(cls, cycle, out_dir).iter_remote_files(), None) is not None:
            return cycle
    return None


def probe_url(url: str, session: requests.Session) -> FileProbe:
    timeout = (CONNECT_TIMEOUT, READ_TIMEOUT)
    try:
        head = session.head(url, timeout=timeout, allow_redirects=True, headers=REQUEST_HEADERS)
    except requests.RequestException as exc:
        return FileProbe(url, False, detail=f"HEAD failed: {exc}")
    size = int(head.headers["Content-Length"]) if "Content-Length" in head.headers else None
    if head.status_code != 200:
        return FileProbe(url, False, head.status_code, size, f"HEAD returned {head.status_code}")
    try:
        resp = session.get(url, headers={**REQUEST_HEADERS, "Range": f"bytes=0-{SNIFF_BYTES - 1}"},
                           timeout=timeout, stream=True)
        data = resp.raw.read(SNIFF_BYTES)
        resp.close()
    except (requests.RequestException, urllib3.exceptions.HTTPError) as exc:
        return FileProbe(url, False, head.status_code, size, f"ranged GET failed: {exc}")
    problem = content_signature_problem(url, data, require_edition2=True)
    return FileProbe(url, not problem, head.status_code, size, problem or "GRIB2")


def sample(urls: list[str]) -> list[str]:
    """First, middle and last entries, without duplicates."""
    picks = [urls[0], urls[len(urls) // 2], urls[-1]] if urls else []
    return list(dict.fromkeys(picks))


def probe_script(script: str, *, probe_all: bool = False, now: datetime | None = None,
                 regs: dict[str, list[str]] | None = None) -> ScriptProbe:
    import forcing_extraction as fe

    result = ScriptProbe(script, registered_as=(regs or {}).get(script, []))
    now = now or datetime.now(timezone.utc)
    with tempfile.TemporaryDirectory() as out_dir:
        try:
            cls = fe.load_downloader_class(SCRIPT_DIR / script)
            result.downloader = cls.__name__
            age = AGE_HOURS_OVERRIDE.get(cls.__name__, DEFAULT_AGE_HOURS)
            cycle = pick_cycle(cls, now, age, out_dir)
            if cycle is None:
                result.error = f"no cycle in the {SEARCH_HOURS} h before now-{age} h yields any files"
                return result
            result.cycle = cycle.strftime("%Y-%m-%d %HZ")
            urls = list(dict.fromkeys(url for _, url, _ in
                                      make_probe_downloader(cls, cycle, out_dir).iter_remote_files()))
        except Exception as exc:  # report and continue with the other scripts
            result.error = f"{type(exc).__name__}: {exc}"
            return result
    result.files_for_cycle = len(urls)
    with requests.Session() as session:
        result.probes = [probe_url(u, session) for u in (urls if probe_all else sample(urls))]
    return result


def format_report(results: list[ScriptProbe]) -> str:
    lines = []
    for r in results:
        status = "OK  " if r.ok else "FAIL"
        reg = ", ".join(r.registered_as) or "not registered"
        lines.append(f"{status} {r.script} [{r.downloader or '?'}] ({reg})")
        if r.error:
            lines.append(f"       {r.error}")
            continue
        good = sum(p.ok for p in r.probes)
        lines.append(f"       cycle {r.cycle}: {r.files_for_cycle} file(s); {good}/{len(r.probes)} probed OK")
        for p in r.probes:
            if not p.ok:
                lines.append(f"       - {p.detail}: {p.url}")
    failed = sum(not r.ok for r in results)
    lines.append(f"\n{len(results) - failed}/{len(results)} downloader script(s) passed.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("scripts", nargs="*", help="Scripts relative to Forcing_Extraction_Scripts (default: all).")
    parser.add_argument("--all", action="store_true", help="Probe every file of the cycle, not a sample of three.")
    parser.add_argument("--json", type=Path, help="Also write results as JSON to this path.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    regs = registrations()
    results = [probe_script(s, probe_all=args.all, regs=regs) for s in (args.scripts or all_scripts())]
    print(format_report(results))
    if args.json:
        args.json.write_text(json.dumps(
            [{**asdict(r), "ok": r.ok} for r in results], indent=2))
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
