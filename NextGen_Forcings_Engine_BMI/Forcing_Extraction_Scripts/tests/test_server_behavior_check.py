"""Offline tests for check_server_behavior: each assumption is detected when a server breaks it."""

import gzip
import hashlib

import pytest

from conftest import FakeResponse
from Forcing_Extraction_Scripts import check_server_behavior as csb

URL = "https://example.test/hrrr.t12z.wrfsfcf01.grib2"
BODY = b"GRIB\x00\x00\x00\x02" + bytes(range(200))
MD5_ETAG = f'"{hashlib.md5(BODY).hexdigest()}"'
LM = "Tue, 10 Nov 2026 12:55:00 GMT"


class FakeSession:
    """Serves one URL; each attribute can be changed to break one assumption."""

    def __init__(self, body=BODY, url=URL):
        self.url = url
        self.body = body
        self.head_headers = {"ETag": MD5_ETAG, "Last-Modified": LM, "Content-Length": str(len(body))}
        self.get_headers = {"ETag": MD5_ETAG, "Last-Modified": LM}
        self.get_length = None
        self.gzip_when_accepted = False

    def head(self, url, headers=None, timeout=None, allow_redirects=True):
        assert timeout is not None
        return FakeResponse(url, 200, b"", self.head_headers, declared_length=int(self.head_headers["Content-Length"]))

    def get(self, url, headers=None, timeout=None, stream=False):
        assert timeout is not None
        if "gzip" in (headers or {}).get("Accept-Encoding", "") and self.gzip_when_accepted:
            encoded = gzip.compress(self.body)
            return FakeResponse(url, 200, encoded, {**self.get_headers, "Content-Encoding": "gzip"})
        return FakeResponse(url, 200, self.body, self.get_headers, self.get_length)


def statuses(report):
    return {c.name: c.status for c in report.checks}


def test_consistent_server_passes():
    report = csb.check_url(URL, FakeSession(), max_bytes=1 << 20)
    assert not report.failed
    assert statuses(report) == {"head": "pass", "compression": "note", "get": "pass", "length": "pass",
                                "validators": "pass", "etag-md5": "pass", "content": "pass"}


@pytest.mark.parametrize("breakage,check", [
    (lambda s: s.head_headers.update({"Content-Length": "999"}), "length"),
    (lambda s: setattr(s, "get_length", 5), "length"),
    (lambda s: s.get_headers.update({"ETag": '"0123456789abcdef0123456789abcdef"'}), "validators"),
    (lambda s: (s.get_headers.update({"ETag": '"0123456789abcdef0123456789abcdef"'}),
                s.head_headers.update({"ETag": '"0123456789abcdef0123456789abcdef"'})), "etag-md5"),
    (lambda s: s.get_headers.update({"Last-Modified": "Wed, 11 Nov 2026 00:00:00 GMT"}), "validators"),
    (lambda s: s.get_headers.update({"Content-Encoding": "gzip"}), "get"),
])
def test_broken_assumption_is_reported(breakage, check):
    session = FakeSession()
    breakage(session)
    report = csb.check_url(URL, session, max_bytes=1 << 20)
    assert statuses(report)[check] == "fail"


def test_non_grib_content_is_reported():
    session = FakeSession(body=b"<html>maintenance</html>")
    session.get_headers["ETag"] = f'"{hashlib.md5(session.body).hexdigest()}"'
    report = csb.check_url(URL, session, max_bytes=1 << 20)
    assert statuses(report)["content"] == "fail"


def test_compression_offered_is_noted_with_saving():
    session = FakeSession(body=b"GRIB\x00\x00\x00\x02" + b"\x00" * 10000)
    session.head_headers["Content-Length"] = str(len(session.body))
    session.get_headers["ETag"] = session.head_headers["ETag"] = f'"{hashlib.md5(session.body).hexdigest()}"'
    session.gzip_when_accepted = True
    report = csb.check_url(URL, session, max_bytes=1 << 20)
    note = next(c for c in report.checks if c.name == "compression")
    assert note.status == "note" and "gzip" in note.detail and "% smaller" in note.detail


def test_large_files_get_header_checks_only():
    report = csb.check_url(URL, FakeSession(), max_bytes=10)
    assert statuses(report)["head"] == "pass" and statuses(report)["length"] == "skip"
