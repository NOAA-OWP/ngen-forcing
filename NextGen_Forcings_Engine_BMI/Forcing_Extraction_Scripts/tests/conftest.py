import io
import sys
from pathlib import Path

import pytest
import requests
import urllib3

BMI_DIR = Path(__file__).resolve().parents[2]
if str(BMI_DIR) not in sys.path:
    sys.path.insert(0, str(BMI_DIR))


def pytest_addoption(parser):
    parser.addoption("--live", action="store_true", default=False,
                     help="Run tests that contact upstream data servers.")


def pytest_configure(config):
    config.addinivalue_line("markers", "live: contacts upstream data servers; enabled with --live")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--live"):
        return
    skip = pytest.mark.skip(reason="contacts upstream servers; run with --live")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)


class FakeRaw:
    def __init__(self, data: bytes, fail_after: int | None = None):
        self._buf = io.BytesIO(data)
        self._fail_after = fail_after

    def read(self, n=-1):
        if self._fail_after is not None and self._buf.tell() >= self._fail_after:
            raise urllib3.exceptions.ProtocolError("Connection broken: simulated")
        if self._fail_after is not None:
            n = min(n if n >= 0 else self._fail_after, self._fail_after - self._buf.tell())
        return self._buf.read(n)


class FakeResponse:
    def __init__(self, url, status=200, data=b"", headers=None, declared_length=None, fail_after=None):
        self.url = url
        self.status_code = status
        self.raw = FakeRaw(data, fail_after)
        self.headers = requests.structures.CaseInsensitiveDict(headers or {})
        self.headers.setdefault("Content-Length", str(len(data) if declared_length is None else declared_length))

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code), response=self)

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeServer:
    """In-memory stand-in for the upstream HTTP servers."""

    def __init__(self):
        self.files = {}
        self.failures = {}
        self.serve_all = False
        self.gets = []
        self.heads = []

    def add(self, url, data=b"GRIB\x00\x00\x00\x02payload", headers=None, declared_length=None,
            fail_times=0, fail_after=None):
        self.files[url] = (data, headers or {}, declared_length)
        if fail_times:
            self.failures[url] = [fail_times, fail_after]

    def _lookup(self, url):
        if url in self.files:
            return self.files[url]
        if self.serve_all:
            return (b"GRIB\x00\x00\x00\x02payload", {}, None)
        return None

    def get(self, url, stream=False, timeout=None, headers=None):
        assert timeout is not None, "requests must always use a timeout"
        self.gets.append(url)
        entry = self._lookup(url)
        if entry is None:
            return FakeResponse(url, 404)
        data, hdrs, declared = entry
        fail_after = None
        if self.failures.get(url, [0])[0] > 0:
            self.failures[url][0] -= 1
            fail_after = self.failures[url][1]
        return FakeResponse(url, 200, data, hdrs, declared, fail_after)

    def head(self, url, timeout=None, **_):
        assert timeout is not None, "requests must always use a timeout"
        self.heads.append(url)
        return FakeResponse(url, 200 if self._lookup(url) else 404)


@pytest.fixture
def server(monkeypatch):
    fake = FakeServer()
    monkeypatch.setattr(requests, "get", fake.get)
    monkeypatch.setattr(requests, "head", fake.head)
    monkeypatch.setattr("time.sleep", lambda s: None)
    return fake
