"""Offline tests for check_server_behavior: each assumption is detected when a server breaks it."""

import gzip
import hashlib

import pytest

from conftest import FakeRaw, FakeResponse, FakeServer
from Forcing_Extraction_Scripts import check_server_behavior as csb

URL = "https://example.test/hrrr.t12z.wrfsfcf01.grib2"
BODY = b"GRIB\x00\x00\x00\x02" + bytes(range(200))
MD5_ETAG = f'"{hashlib.md5(BODY).hexdigest()}"'
LM = "Tue, 10 Nov 2026 12:55:00 GMT"


class FakeSession(FakeServer):
    """Serves one URL like FakeServer; each attribute can be changed to break one assumption."""

    def __init__(self, body=BODY, url=URL):
        super().__init__()
        self.url = url
        self.body = body
        self.head_headers = {"ETag": MD5_ETAG, "Last-Modified": LM, "Content-Length": str(len(body))}
        self.get_headers = {"ETag": MD5_ETAG, "Last-Modified": LM}
        self.get_length = None
        self.gzip_when_accepted = False
        self.honor_conditions = True
        self.range_shift = 0

    def head(self, url, headers=None, timeout=None, allow_redirects=True):
        assert timeout is not None
        return FakeResponse(url, 200, b"", self.head_headers, declared_length=int(self.head_headers["Content-Length"]))

    def get(self, url, headers=None, timeout=None, stream=False):
        headers = dict(headers or {})
        if self.gzip_when_accepted and "gzip" in headers.get("Accept-Encoding", ""):
            return FakeResponse(url, 200, gzip.compress(self.body), {**self.get_headers, "Content-Encoding": "gzip"})
        if not self.honor_conditions:
            headers.pop("If-None-Match", None)
            headers.pop("If-Modified-Since", None)
        self.add(url, self.body, headers=self.get_headers, declared_length=self.get_length)
        response = super().get(url, stream=stream, timeout=timeout, headers=headers)
        if response.status_code == 206 and self.range_shift:
            response.raw = FakeRaw(response.raw.read() + b"\x00" * self.range_shift)
        return response


def statuses(report):
    return {c.name: c.status for c in report.checks}


def test_consistent_server_passes():
    report = csb.check_url(URL, FakeSession(), max_bytes=1 << 20)
    assert not report.failed
    assert statuses(report) == {"head": "pass", "compression": "note", "get": "pass", "length": "pass",
                                "validators": "pass", "etag-md5": "pass", "content": "pass",
                                "conditional-etag": "pass", "conditional-date": "pass", "range": "pass",
                                "if-range": "pass"}


@pytest.mark.parametrize("breakage,check", [
    (lambda s: s.head_headers.update({"Content-Length": "999"}), "length"),
    (lambda s: setattr(s, "get_length", 5), "length"),
    (lambda s: s.get_headers.update({"ETag": '"0123456789abcdef0123456789abcdef"'}), "validators"),
    (lambda s: (s.get_headers.update({"ETag": '"0123456789abcdef0123456789abcdef"'}),
                s.head_headers.update({"ETag": '"0123456789abcdef0123456789abcdef"'})), "etag-md5"),
    (lambda s: s.get_headers.update({"Last-Modified": "Wed, 11 Nov 2026 00:00:00 GMT"}), "validators"),
    (lambda s: s.get_headers.update({"Content-Encoding": "gzip"}), "get"),
    (lambda s: setattr(s, "honor_conditions", False), "conditional-etag"),
    (lambda s: setattr(s, "honor_conditions", False), "conditional-date"),
    (lambda s: setattr(s, "range_shift", 1), "range"),
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


def test_multipart_etag_is_reproduced():
    body = BODY
    digests = b"".join(hashlib.md5(body[i:i + 52]).digest() for i in range(0, len(body), 52))
    etag = f'"{hashlib.md5(digests).hexdigest()}-4"'
    session = FakeSession()
    session.head_headers["ETag"] = session.get_headers["ETag"] = etag
    report = csb.check_url(URL, session, max_bytes=1 << 20)
    check = next(c for c in report.checks if c.name == "etag-md5")
    assert check.status == "pass" and "multipart" in check.detail


def test_large_files_get_header_checks_only():
    report = csb.check_url(URL, FakeSession(), max_bytes=10)
    assert statuses(report)["head"] == "pass" and statuses(report)["length"] == "skip"
