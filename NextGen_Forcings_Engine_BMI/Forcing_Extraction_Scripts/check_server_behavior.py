"""
Check, against the live servers, the assumptions the downloaders make about server metadata.

For each source and region a recently published cycle is chosen (as in probe_sources), the files
the downloader would fetch are listed, and a sample of them is checked. Each check passes, fails
(the server contradicts an assumption), or is skipped (not applicable or not checkable):

  head        HEAD answers 200 with a Content-Length.
  get         GET answers 200 without a transfer Content-Encoding (Accept-Encoding: identity).
  length      HEAD Content-Length, GET Content-Length and the bytes received agree.
  validators  GET reports the same ETag and Last-Modified as HEAD.
  etag-md5    an ETag of 32 hex digits is the MD5 of the bytes received; a multipart ETag
              ("<md5>-N") is reproduced from the bytes when its part size can be inferred.
  content     the bytes are what the name says (GRIB2, or gzip containing GRIB2).
  conditional-etag  If-None-Match with the current ETag answers 304 carrying that ETag, and with
              another ETag answers 200.
  conditional-date  If-Modified-Since with Last-Modified answers 304, with an earlier date 200.
  range       a byte range answers 206 with a matching Content-Range and exactly those bytes.
  if-range    Range with If-Range of the current ETag answers 206; with another ETag, 200.
  compression (note) whether the server would compress the transfer if gzip were accepted, and
              how much smaller the body would be. The downloaders request identity encoding;
              a server that starts compressing is a reason to revisit that.

Files larger than --max-mb get the HEAD check only. Run from NextGen_Forcings_Engine_BMI/:

    python -m Forcing_Extraction_Scripts.check_server_behavior                 # all sources
    python -m Forcing_Extraction_Scripts.check_server_behavior HRRR[CONUS] --max-mb 200
    python -m Forcing_Extraction_Scripts.check_server_behavior --url https://host/path/file.grib2
"""

from __future__ import annotations

import argparse
import email.utils
import hashlib
import io
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlparse

import requests
import urllib3

from Forcing_Extraction_Scripts import probe_sources
from Forcing_Extraction_Scripts.forecast_download_base import (
    CHUNK_SIZE,
    REQUEST_HEADERS,
    SNIFF_BYTES,
    content_signature_problem,
    etag_matches,
    plain_md5_etag,
)
from Forcing_Extraction_Scripts.probe_sources import TIMEOUT, get_prefix
from Forcing_Extraction_Scripts.sources import Source

PASS, FAIL, SKIP, NOTE = "pass", "fail", "skip", "note"
DEFAULT_MAX_MB = 64


@dataclass
class Check:
    name: str
    status: str
    detail: str = ""


@dataclass
class FileReport:
    url: str
    size: int | None = None
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str = "") -> None:
        self.checks.append(Check(name, status, detail))

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if c.status == FAIL]


@dataclass
class Fetched:
    """What a full GET returned: headers, byte count, MD5 and leading bytes."""

    status: int
    headers: dict[str, str]
    length: int = 0
    md5: str = ""
    head: bytes = b""
    body: bytes = b""


def fetch_all(session: requests.Session, url: str, keep_body: bool = False) -> Fetched:
    with session.get(url, headers=REQUEST_HEADERS, timeout=TIMEOUT, stream=True) as resp:
        result = Fetched(resp.status_code, dict(resp.headers))
        if resp.status_code != 200:
            return result
        digest = hashlib.md5()
        chunks = []
        while True:
            chunk = resp.raw.read(CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
            if len(result.head) < SNIFF_BYTES:
                result.head += chunk[:SNIFF_BYTES - len(result.head)]
            if keep_body:
                chunks.append(chunk)
            result.length += len(chunk)
        result.md5 = digest.hexdigest()
        result.body = b"".join(chunks)
    return result


def check_url(url: str, session: requests.Session, max_bytes: int) -> FileReport:
    report = FileReport(url)
    try:
        head = probe_sources.head(session, url)
    except requests.RequestException as exc:
        report.add("head", FAIL, f"HEAD failed: {exc}")
        return report
    if head.status_code != 200:
        report.add("head", FAIL, f"HEAD returned {head.status_code}")
        return report
    head_length = head.headers.get("Content-Length")
    if head_length is None:
        report.add("head", FAIL, "HEAD has no Content-Length")
        return report
    report.size = int(head_length)
    report.add("head", PASS, f"{report.size} bytes, ETag {head.headers.get('ETag')}, "
                             f"Last-Modified {head.headers.get('Last-Modified')}")
    check_compression(report, url, session)
    if report.size > max_bytes:
        for name in ("get", "length", "validators", "etag-md5", "content",
                     "conditional-etag", "conditional-date", "range", "if-range"):
            report.add(name, SKIP, f"larger than {max_bytes // (1 << 20)} MB")
        return report

    try:
        got = fetch_all(session, url, keep_body=True)
    except (requests.RequestException, urllib3.exceptions.HTTPError) as exc:
        report.add("get", FAIL, f"GET failed: {exc}")
        return report
    encoding = got.headers.get("Content-Encoding", "identity").lower()
    if got.status != 200:
        report.add("get", FAIL, f"GET returned {got.status}")
        return report
    if encoding not in ("", "identity") and not url.endswith(".gz"):
        report.add("get", FAIL, f"Content-Encoding {encoding} despite Accept-Encoding: identity")
    else:
        report.add("get", PASS, f"Content-Encoding {encoding}")

    get_length = got.headers.get("Content-Length")
    lengths = {"HEAD": report.size, "GET": int(get_length) if get_length else None, "received": got.length}
    if len({v for v in lengths.values() if v is not None}) == 1 and get_length is not None:
        report.add("length", PASS, f"{got.length} bytes")
    else:
        report.add("length", FAIL, ", ".join(f"{k} {v}" for k, v in lengths.items()))

    mismatched = [h for h in ("ETag", "Last-Modified") if head.headers.get(h) != got.headers.get(h)]
    if mismatched:
        report.add("validators", FAIL, "; ".join(
            f"{h}: HEAD {head.headers.get(h)!r}, GET {got.headers.get(h)!r}" for h in mismatched))
    else:
        report.add("validators", PASS)

    etag = got.headers.get("ETag")
    verdict = etag_matches(io.BytesIO(got.body), etag)
    if verdict is True:
        report.add("etag-md5", PASS, "multipart ETag" if plain_md5_etag(etag) is None else "")
    elif verdict is False:
        report.add("etag-md5", FAIL, f"ETag {etag} but content MD5 {got.md5}")
    else:
        report.add("etag-md5", SKIP, f"ETag {etag!r} is not an MD5, or a multipart ETag not reproducible")

    problem = content_signature_problem(url, got.head, require_edition2=True)
    report.add("content", FAIL if problem else PASS, problem)
    extra_checks(report, url, session, head, got)
    return report


def check_compression(report: FileReport, url: str, session: requests.Session) -> None:
    """Note whether the server compresses the transfer when gzip is accepted (headers only)."""
    try:
        _, headers, _ = get_prefix(session, url, {"Accept-Encoding": "gzip, deflate"})
        encoding = headers.get("Content-Encoding", "identity").lower()
        length = headers.get("Content-Length")
    except requests.RequestException as exc:
        report.add("compression", SKIP, f"GET with gzip accepted failed: {exc}")
        return
    if encoding in ("", "identity") or (url.endswith(".gz") and encoding == "gzip"):
        report.add("compression", NOTE, "not compressed when gzip is accepted")
    elif length and report.size:
        saving = 100 * (report.size - int(length)) / report.size
        report.add("compression", NOTE, f"{encoding}: {length} of {report.size} bytes ({saving:.0f}% smaller)")
    else:
        report.add("compression", NOTE, f"{encoding} (encoded length not given)")


OTHER_ETAG = '"00000000000000000000000000000000"'


def extra_checks(report: FileReport, url: str, session: requests.Session,
                 head: requests.Response, got: Fetched) -> None:
    """Conditional-request and range behavior the validator logic relies on."""
    etag, last_modified, size = got.headers.get("ETag"), got.headers.get("Last-Modified"), got.length
    try:
        if etag:
            same, same_headers, _ = get_prefix(session, url, {"If-None-Match": etag})
            other, _, _ = get_prefix(session, url, {"If-None-Match": OTHER_ETAG})
            echoed = same_headers.get("ETag", etag)
            if same == 304 and echoed == etag and other == 200:
                report.add("conditional-etag", PASS)
            else:
                report.add("conditional-etag", FAIL, f"current ETag -> {same} (ETag {echoed}), other -> {other}")
        else:
            report.add("conditional-etag", SKIP, "no ETag")

        if last_modified:
            earlier = email.utils.format_datetime(
                email.utils.parsedate_to_datetime(last_modified) - timedelta(days=1), usegmt=True)
            same, _, _ = get_prefix(session, url, {"If-Modified-Since": last_modified})
            older, _, _ = get_prefix(session, url, {"If-Modified-Since": earlier})
            if same == 304 and older == 200:
                report.add("conditional-date", PASS)
            else:
                report.add("conditional-date", FAIL, f"Last-Modified -> {same}, a day earlier -> {older}")
        else:
            report.add("conditional-date", SKIP, "no Last-Modified")

        if size < 32:
            report.add("range", SKIP, "file too small")
            report.add("if-range", SKIP, "file too small")
            return
        start, end = size // 3, size // 3 + 15
        status, headers, body = get_prefix(session, url, {"Range": f"bytes={start}-{end}"}, read=64)
        wanted = f"bytes {start}-{end}/{size}"
        if status == 206 and headers.get("Content-Range") == wanted and body == got.body[start:end + 1]:
            report.add("range", PASS)
        else:
            report.add("range", FAIL, f"{status}, Content-Range {headers.get('Content-Range')!r} (wanted {wanted!r}), "
                                      f"{'matching' if body == got.body[start:end + 1] else 'different'} bytes")

        if etag:
            matching, _, _ = get_prefix(session, url, {"Range": f"bytes={start}-{end}", "If-Range": etag})
            stale, _, _ = get_prefix(session, url, {"Range": f"bytes={start}-{end}", "If-Range": OTHER_ETAG})
            if matching == 206 and stale == 200:
                report.add("if-range", PASS)
            else:
                report.add("if-range", FAIL, f"current ETag -> {matching}, other -> {stale}")
        else:
            report.add("if-range", SKIP, "no ETag")
    except (requests.RequestException, urllib3.exceptions.HTTPError) as exc:
        report.add("conditional/range", FAIL, f"request failed: {exc}")


@dataclass
class SourceReport:
    source: str
    cycle: str = ""
    files: list[FileReport] = field(default_factory=list)
    error: str = ""


def check_source(source: Source, session: requests.Session, max_bytes: int, per_source: int) -> SourceReport:
    result = SourceReport(probe_sources.source_label(source))
    try:
        chosen = probe_sources.source_cycle(source)
    except Exception as exc:  # report and continue with the other sources
        result.error = f"{type(exc).__name__}: {exc}"
        return result
    result.cycle = chosen.cycle.strftime("%Y-%m-%d %HZ") + (f" ({chosen.note})" if chosen.note else "")
    for url in probe_sources.sample(chosen.urls)[:per_source]:
        result.files.append(check_url(url, session, max_bytes))
    return result


def summarize(files: list[FileReport]) -> str:
    lines = ["", "Assumption summary (pass / fail / skip), by host:"]
    by_host: dict[str, Counter] = {}
    for f in files:
        host = urlparse(f.url).netloc
        for c in f.checks:
            by_host.setdefault(host, Counter())[(c.name, c.status)] += 1
    for host, counts in sorted(by_host.items()):
        names = sorted({n for n, _ in counts})
        lines.append(f"  {host}")
        for n in names:
            notes = f"   ({counts[(n, NOTE)]} note)" if counts[(n, NOTE)] else ""
            lines.append(f"    {n:<11} {counts[(n, PASS)]:>3} / {counts[(n, FAIL)]:>3} / {counts[(n, SKIP)]:>3}{notes}")
    return "\n".join(lines)


def format_file(f: FileReport, indent: str = "  ") -> list[str]:
    status = "FAIL" if f.failed else "ok  "
    lines = [f"{indent}{status} {f.url}"]
    for c in f.checks:
        if c.status in (FAIL, SKIP) or (c.status == NOTE and not c.detail.startswith("not compressed")):
            lines.append(f"{indent}       {c.status.upper()} {c.name}: {c.detail}")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("sources", nargs="*",
                        help="Sources as NAME (every region) or NAME[Region], e.g. HRRR or NBM[Alaska] (default: all).")
    parser.add_argument("--url", action="append", default=[], help="Check this URL (repeatable) instead of sources.")
    parser.add_argument("--max-mb", type=int, default=DEFAULT_MAX_MB,
                        help=f"Download files up to this size for full checks (default {DEFAULT_MAX_MB}).")
    parser.add_argument("--per-source", type=int, default=3, help="Files sampled per source (default 3).")
    parser.add_argument("--json", type=Path, help="Also write results as JSON to this path.")
    args = parser.parse_args(argv)
    max_bytes = args.max_mb << 20
    try:
        sources = probe_sources.sources_from_args(args.sources)
    except ValueError as exc:
        parser.error(str(exc))

    lines: list[str] = []
    files: list[FileReport] = []
    reports: list[SourceReport] = []
    with probe_sources.PoliteSession() as session:
        if args.url:
            for url in args.url:
                f = check_url(url, session, max_bytes)
                files.append(f)
                lines += format_file(f, "")
        else:
            for source in sources:
                r = check_source(source, session, max_bytes, args.per_source)
                reports.append(r)
                files += r.files
                lines.append(f"{r.source} ({r.cycle or r.error})")
                for f in r.files:
                    lines += format_file(f)
    print("\n".join(lines) + summarize(files))
    if args.json:
        args.json.write_text(json.dumps({"files": [asdict(f) for f in files],
                                         "sources": [asdict(r) for r in reports]}, indent=2))
    return 1 if any(f.failed for f in files) else 0


if __name__ == "__main__":
    sys.exit(main())
