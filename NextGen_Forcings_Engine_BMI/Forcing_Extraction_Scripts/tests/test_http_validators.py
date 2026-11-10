"""Offline tests for HTTP validator handling: sidecars, conditional requests, resume and checks."""

import datetime as dt
import hashlib
import logging
import os
from email.utils import parsedate_to_datetime

import pytest

from conftest import FakeResponse
from Forcing_Extraction_Scripts import forecast_download_base as fdb
from test_downloaders import CYCLE, HRRR_ANA, HRRR_ANA_URL, NDFD, load, make, serve_ndfd

DATA = b"GRIB-0123456789abcdef"
ETAG = f'"{hashlib.md5(DATA).hexdigest()}"'
LM = "Tue, 10 Nov 2026 12:55:00 GMT"
S3_HEADERS = {"ETag": ETAG, "Last-Modified": LM, "Accept-Ranges": "bytes"}
HRRR_PATH = "hrrr.20261110/conus/hrrr.t12z.wrfsfcf01.grib2"

STAGE4 = "CONUS/get_conus_StageIV.py"
STAGE4_URL = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/pcpanl/v4.1/pcpanl.20261110/st4_conus.2026111012.01h.grb2"


@pytest.fixture(autouse=True)
def fixed_now(monkeypatch):
    """Pin "now" shortly after CYCLE so archived-cycle logic is deterministic."""
    monkeypatch.setattr(fdb, "utc_now", lambda: CYCLE + dt.timedelta(hours=3))


def test_sidecar_records_validators_and_mtime(server, tmp_path):
    server.add(HRRR_ANA_URL, DATA, headers=S3_HEADERS)
    make(HRRR_ANA, tmp_path).run()
    path = str(tmp_path / HRRR_PATH)
    meta = fdb.read_meta(path)
    assert meta["url"] == HRRR_ANA_URL
    assert meta["etag"] == ETAG and meta["last_modified"] == LM
    assert meta["content_length"] == len(DATA) and meta["md5_verified"] is True
    assert os.path.getmtime(path) == parsedate_to_datetime(LM).timestamp()


def test_immutable_existing_file_is_not_requested_again(server, tmp_path):
    server.add(HRRR_ANA_URL, DATA, headers=S3_HEADERS)
    make(HRRR_ANA, tmp_path).run()
    make(HRRR_ANA, tmp_path).run()
    assert server.gets == [HRRR_ANA_URL]


def test_immutable_file_with_wrong_size_is_fetched_again(server, tmp_path):
    server.add(HRRR_ANA_URL, DATA, headers=S3_HEADERS)
    make(HRRR_ANA, tmp_path).run()
    (tmp_path / HRRR_PATH).write_bytes(DATA[:5])
    make(HRRR_ANA, tmp_path).run()
    assert len(server.gets) == 2
    assert "If-None-Match" not in server.request_headers[-1]
    assert (tmp_path / HRRR_PATH).read_bytes() == DATA


def test_revalidate_sends_conditional_request(server, tmp_path):
    server.add(HRRR_ANA_URL, DATA, headers=S3_HEADERS)
    make(HRRR_ANA, tmp_path).run()
    make(HRRR_ANA, tmp_path, revalidate=True).run()
    assert server.request_headers[-1]["If-None-Match"] == ETAG
    assert "Cache-Control" not in server.request_headers[-1]  # immutable product
    assert (tmp_path / HRRR_PATH).read_bytes() == DATA


def test_mutable_product_is_updated_when_changed_upstream(server, tmp_path, caplog):
    caplog.set_level(logging.INFO)
    server.add(STAGE4_URL, b"GRIB first", headers={"Last-Modified": "Tue, 10 Nov 2026 12:56:00 GMT"})
    make(STAGE4, tmp_path).run()
    make(STAGE4, tmp_path).run()  # unchanged: 304
    assert server.request_headers[-1]["If-Modified-Since"] == "Tue, 10 Nov 2026 12:56:00 GMT"
    assert server.request_headers[-1]["Cache-Control"] == "no-cache"
    assert (tmp_path / "st4_conus.2026111012.01h.grb2").read_bytes() == b"GRIB first"

    server.add(STAGE4_URL, b"GRIB rerun", headers={"Last-Modified": "Wed, 11 Nov 2026 12:56:00 GMT"})
    make(STAGE4, tmp_path).run()
    assert (tmp_path / "st4_conus.2026111012.01h.grb2").read_bytes() == b"GRIB rerun"
    assert fdb.read_meta(str(tmp_path / "st4_conus.2026111012.01h.grb2"))["last_modified"].startswith("Wed")
    assert "Updated" in caplog.text


def test_failed_revalidation_keeps_existing_file(server, tmp_path, caplog):
    server.add(STAGE4_URL, b"GRIB first", headers={"Last-Modified": LM})
    make(STAGE4, tmp_path).run()
    del server.files[STAGE4_URL]
    assert make(STAGE4, tmp_path).run() is True
    assert (tmp_path / "st4_conus.2026111012.01h.grb2").read_bytes() == b"GRIB first"
    assert "keeping the existing file" in caplog.text


def test_304_to_unconditional_request_is_reported(server, tmp_path, monkeypatch, caplog):
    monkeypatch.setattr("requests.get", lambda url, **kw: FakeResponse(url, 304))
    with pytest.raises(fdb.DownloadError, match="304 Not Modified to a request without conditions"):
        make(HRRR_ANA, tmp_path).run()
    assert not (tmp_path / HRRR_PATH).exists()


def test_304_with_a_different_etag_is_reported_and_refetched(server, tmp_path, monkeypatch, caplog):
    server.add(HRRR_ANA_URL, DATA, headers=S3_HEADERS)
    make(HRRR_ANA, tmp_path).run()
    new_data = b"GRIB-new-content"
    server.add(HRRR_ANA_URL, new_data, headers={"ETag": f'"{hashlib.md5(new_data).hexdigest()}"'})
    original_get = server.get

    def get(url, **kw):
        if (kw.get("headers") or {}).get("If-None-Match"):
            return FakeResponse(url, 304, b"", {"ETag": '"something-else"'})
        return original_get(url, **kw)

    monkeypatch.setattr("requests.get", get)
    make(HRRR_ANA, tmp_path, revalidate=True, max_download_attempts=2).run()
    assert (tmp_path / HRRR_PATH).read_bytes() == new_data
    assert any(r.levelname == "ERROR" and "304 Not Modified with ETag" in r.message for r in caplog.records)


def test_md5_mismatch_is_rejected(server, tmp_path, caplog):
    server.add(HRRR_ANA_URL, DATA, headers={"ETag": f'"{hashlib.md5(b"other").hexdigest()}"'})
    with pytest.raises(fdb.DownloadError, match="MD5 mismatch"):
        make(HRRR_ANA, tmp_path).run()
    assert not (tmp_path / HRRR_PATH).exists()
    assert any(r.levelname == "ERROR" and "inconsistent with its metadata" in r.message for r in caplog.records)


def test_unreproducible_multipart_etag_is_left_unverified(server, tmp_path):
    server.add(HRRR_ANA_URL, DATA, headers={"ETag": '"0123456789abcdef0123456789abcdef-4"'})
    make(HRRR_ANA, tmp_path).run()
    assert fdb.read_meta(str(tmp_path / HRRR_PATH))["md5_verified"] is False


def test_multipart_etag_is_verified_when_part_size_is_inferred(server, tmp_path):
    data = b"GRIB" + bytes(range(26))  # 30 bytes in 3 parts of 10
    digests = b"".join(hashlib.md5(data[i:i + 10]).digest() for i in range(0, 30, 10))
    etag = f'"{hashlib.md5(digests).hexdigest()}-3"'
    server.add(HRRR_ANA_URL, data, headers={"ETag": etag})
    make(HRRR_ANA, tmp_path).run()
    assert fdb.read_meta(str(tmp_path / HRRR_PATH))["md5_verified"] is True


@pytest.mark.parametrize("size,parts,first", [(30, 3, 10), (31, 4, 8), ((1 << 20) * 20, 3, 6990507)])
def test_multipart_part_sizes(size, parts, first):
    sizes = fdb.multipart_part_sizes(size, parts)
    assert sizes[0] == first and all(-(-size // p) == parts for p in sizes)


def test_interrupted_body_resumes_with_range(server, tmp_path):
    server.add(HRRR_ANA_URL, DATA, headers=S3_HEADERS, fail_times=1, fail_after=7)
    make(HRRR_ANA, tmp_path, max_download_attempts=2).run()
    assert server.request_headers[1]["Range"] == "bytes=7-"
    assert server.request_headers[1]["If-Range"] == ETAG
    assert (tmp_path / HRRR_PATH).read_bytes() == DATA


@pytest.mark.parametrize("content_range,match", [
    ("bytes 0-20/21", "asked for bytes 7- but got"),
    ("bytes 7-15/21", "stops before the end"),
    ("bytes 7-20/21", None),
])
def test_resumed_range_is_verified(server, tmp_path, monkeypatch, caplog, content_range, match):
    server.add(HRRR_ANA_URL, DATA, headers=S3_HEADERS, fail_times=1, fail_after=7)
    original_get = server.get

    def get(url, **kw):
        response = original_get(url, **kw)
        if response.status_code == 206:
            response.headers["Content-Range"] = content_range
        return response

    monkeypatch.setattr("requests.get", get)
    make(HRRR_ANA, tmp_path, max_download_attempts=3).run()
    assert (tmp_path / HRRR_PATH).read_bytes() == DATA
    errors = [r.message for r in caplog.records if r.levelname == "ERROR"]
    if match:
        assert any(match in m for m in errors)
        assert "Range" not in server.request_headers[-1]  # the partial file was discarded
    else:
        assert not errors


def test_206_to_request_without_range_is_reported(server, tmp_path, monkeypatch):
    monkeypatch.setattr("requests.get", lambda url, **kw: FakeResponse(
        url, 206, DATA, {"Content-Range": f"bytes 0-{len(DATA) - 1}/{len(DATA)}"}))
    with pytest.raises(fdb.DownloadError, match="206 Partial Content to a request without Range"):
        make(HRRR_ANA, tmp_path).run()


def test_resume_restarts_when_object_changed(server, tmp_path, monkeypatch):
    server.add(HRRR_ANA_URL, DATA, headers=S3_HEADERS, fail_times=1, fail_after=7)
    new_data = b"GRIB-replaced-object"
    original_get = server.get

    def get_then_replace(url, **kw):
        response = original_get(url, **kw)
        server.add(url, new_data, headers={**S3_HEADERS, "ETag": f'"{hashlib.md5(new_data).hexdigest()}"'})
        return response

    monkeypatch.setattr("requests.get", get_then_replace)
    make(HRRR_ANA, tmp_path, max_download_attempts=2).run()
    assert server.request_headers[1]["If-Range"] == ETAG
    assert (tmp_path / HRRR_PATH).read_bytes() == new_data


def test_404_for_archived_cycle_is_not_retried(server, tmp_path, monkeypatch):
    monkeypatch.setattr(fdb, "utc_now", lambda: CYCLE + dt.timedelta(days=5))
    with pytest.raises(fdb.DownloadError, match="not retried"):
        make(HRRR_ANA, tmp_path, max_download_attempts=5).run()
    assert len(server.gets) == 1


def test_404_for_recent_cycle_is_retried(server, tmp_path):
    with pytest.raises(fdb.DownloadError):
        make(HRRR_ANA, tmp_path, max_download_attempts=3).run()
    assert len(server.gets) == 3


def test_fixed_file_cleanup_removes_sidecars(server, tmp_path):
    server.serve_all = True
    flat = tmp_path / "MultiSensor_QPE_01H_Pass1_00.00"
    flat.mkdir()
    old = flat / "MRMS_MultiSensor_QPE_01H_Pass1_00.00_20261109-120000.grib2.gz"
    old.write_bytes(b"old")
    fdb.write_meta(str(old), {"url": "x"})
    make("CONUS/get_conus_MRMS_MultiSensor.py", tmp_path, start=CYCLE, lookback_hours=3, cleanback_hours=240).run()
    assert not old.exists() and not os.path.exists(fdb.meta_path(str(old)))


def test_cli_revalidate_flag(tmp_path):
    downloader = load(HRRR_ANA).from_cli_args(argv=[str(tmp_path), "2026-11-10 13:00:00", "--revalidate"])
    assert downloader.revalidate is True


# ----------------------------------------------------------------------------- NDFD parts

NDFD_OUT = "NDFD/20261110/wgrbbul/ndfd_conus"


def ndfd_headers(tag):
    return {"Last-Modified": "Tue, 10 Nov 2026 12:30:00 GMT", "ETag": f'"{tag}"'}


def test_ndfd_unchanged_parts_leave_file_untouched(server, tmp_path):
    serve_ndfd(server, headers=ndfd_headers("v1"))
    make("CONUS/get_conus_NDFD.py", tmp_path).run()
    tmp_file = tmp_path / NDFD_OUT / "ndfd_conus_tmp.grib2"
    inode = tmp_file.stat().st_ino
    first = len(server.gets)
    make("CONUS/get_conus_NDFD.py", tmp_path).run()
    assert len(server.gets) - first == 7  # one conditional request per part
    assert all(h.get("If-None-Match") == '"v1"' for h in server.request_headers[first:])
    assert tmp_file.stat().st_ino == inode


def test_ndfd_changed_part_rebuilds_field(server, tmp_path):
    serve_ndfd(server, headers=ndfd_headers("v1"))
    make("CONUS/get_conus_NDFD.py", tmp_path).run()
    server.add(f"{NDFD}/VP.004-007/ds.temp.bin", b"GRIB new days 4-7;", headers=ndfd_headers("v2"))
    make("CONUS/get_conus_NDFD.py", tmp_path).run()
    assert (tmp_path / NDFD_OUT / "ndfd_conus_tmp.grib2").read_bytes() == \
        b"GRIB VP.001-003 ds.temp.bin;GRIB new days 4-7;"
    parts = fdb.read_meta(str(tmp_path / NDFD_OUT / "ndfd_conus_tmp.grib2"))["parts"]
    assert parts[f"{NDFD}/VP.004-007/ds.temp.bin"]["etag"] == '"v2"'


def test_ndfd_newer_issuance_does_not_overwrite_cycle(server, tmp_path, caplog):
    serve_ndfd(server, headers=ndfd_headers("v1"))
    make("CONUS/get_conus_NDFD.py", tmp_path).run()
    serve_ndfd(server, headers={"Last-Modified": "Thu, 12 Nov 2026 12:00:00 GMT", "ETag": '"v9"'})
    for bin_file in ("ds.temp.bin",):
        server.add(f"{NDFD}/VP.001-003/{bin_file}", b"GRIB later issuance;",
                   headers={"Last-Modified": "Thu, 12 Nov 2026 12:00:00 GMT", "ETag": '"v9"'})
    assert make("CONUS/get_conus_NDFD.py", tmp_path).run() is True
    assert (tmp_path / NDFD_OUT / "ndfd_conus_tmp.grib2").read_bytes().startswith(b"GRIB VP.001-003")
    assert "keeping the existing file" in caplog.text


# ----------------------------------------------------------------------------- availability check

def check(tmp_path, caplog):
    caplog.clear()
    ok = make(HRRR_ANA, tmp_path, check_file_availability=1).run()
    return ok, caplog.text


@pytest.mark.parametrize("prepare,status,ok", [
    ("none", "available, not downloaded", True),
    ("downloaded", "downloaded and current", True),
    ("changed", "downloaded but changed upstream", True),
    ("truncated", "downloaded but size differs", True),
    ("no-sidecar", "downloaded, no stored validators", True),
    ("missing", "missing upstream", False),
])
def test_availability_check_classifies_files(server, tmp_path, caplog, prepare, status, ok):
    caplog.set_level(logging.INFO)
    server.add(HRRR_ANA_URL, DATA, headers=S3_HEADERS)
    if prepare != "none" and prepare != "missing":
        make(HRRR_ANA, tmp_path).run()
    if prepare == "changed":
        server.add(HRRR_ANA_URL, DATA, headers={**S3_HEADERS, "ETag": '"changed"'})
    elif prepare == "truncated":
        (tmp_path / HRRR_PATH).write_bytes(DATA[:3])
    elif prepare == "no-sidecar":
        os.remove(fdb.meta_path(str(tmp_path / HRRR_PATH)))
    elif prepare == "missing":
        del server.files[HRRR_ANA_URL]
    result, text = check(tmp_path, caplog)
    assert result is ok
    assert f"{status}: 1" in text
