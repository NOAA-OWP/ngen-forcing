#!/usr/bin/env python
# download_noaa_obv_wl.py
# Reads YAML config → passes params to worker (no processing in reader).
# Supports ISO-like timestamps (YYYY-MM-DDTHH-MM-SSZ) in config.

import argparse
import os
import json
import datetime
import urllib.request
import urllib.parse
from typing import List, Dict, Any, Optional
import re
import numpy as np
import xarray as xr


try:
    import yaml  # PyYAML
except Exception as e:
    raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from e


# ---------------------------
# Helper: station discovery
# ---------------------------
import re
from typing import List

def get_waterlevel_station_ids_from_obs(obs_path: str) -> List[int]:
    """
    Extract numeric station IDs (digits inside parentheses) from a sfincs.obs file.
    Example line:
      830344.95 3187383.41 "Sargent (8772985)"
    Only keeps IDs that are purely digits (skips alphanumeric like mg0101).
    """
    ids = []
    with open(obs_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Look for "(digits)" inside quotes
            match = re.search(r"\((\d+)\)", line)
            if match:
                ids.append(int(match.group(1)))
    return ids


def get_waterlevel_station_ids_from_nc(nc_path: str, var_candidates=("station_name", "station_names", "name", "names")) -> List[int]:
    """
    Extract numeric station IDs (digits inside parentheses) from a NetCDF file.
    Only keeps IDs that are purely digits (e.g., (8772985)), skips alphanumeric like (mg0101).
    """
    def _clean(s: str) -> str:
        return s.replace("\x00", " ").strip()

    with xr.open_dataset(nc_path, decode_cf=True) as ds:
        var = next((v for v in var_candidates if v in ds.variables), None)
        if var is None:
            raise KeyError(f"No station-name variable found. Tried: {var_candidates}")

        da = ds[var]

        if da.ndim == 1 and (da.dtype.kind in {"U", "S"} or da.dtype == object):
            raw = da.values
            names = [str(x) for x in raw.tolist()]
        elif da.ndim == 2:
            arr = da.values  # (stations, name_len)
            if arr.dtype.kind == "S":  # bytes
                arr = np.char.decode(arr, "utf-8", errors="ignore")
            names = ["".join(row.tolist()) for row in arr]
        else:
            raise TypeError(f"Unsupported station-name array shape/dtype: shape={da.shape}, dtype={da.dtype}")

    # Extract numeric IDs inside parentheses
    ids = []
    for s in names:
        s = _clean(s)
        match = re.search(r"\((\d+)\)", s)
        if match:
            ids.append(int(match.group(1)))

    return ids


def find_all_waterlevel_station_ids(
    site_noL: List[str],
    *,
    station_type: str = "historicwl",
    base_url: str = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json",
    extra_params: Optional[Dict[str, Any]] = None,
) -> List[str]:
    params = {"type": station_type}
    if extra_params:
        params.update({k: v for k, v in extra_params.items() if v is not None})

    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    print("[INFO] Fetching water level stations:", url)

    try:
        with urllib.request.urlopen(url) as rno:
            payload = rno.read().decode("utf-8")
        json_data = json.loads(payload)
    except Exception as e:
        print("[ERROR] Station discovery failed:", e)
        return site_noL

    count = int(json_data.get("count", 0))
    print(f"[INFO] MDAPI reported {count} stations for type='{station_type}'")

    for sta in json_data.get("stations", []):
        sid = sta.get("id")
        if sid:
            site_noL.append(sid)

    return site_noL


# ---------------------------
# Date parsing utility
# ---------------------------

def parse_config_datetime(timestr: str) -> datetime.datetime:
    """
    Parse config timestamps like '2020-05-11T00-00-00Z' into naive UTC datetime.
    """
    try:
        return datetime.datetime.strptime(timestr, "%Y-%m-%dT%H-%M-%SZ")
    except ValueError:
        raise ValueError(f"Invalid time format in config: {timestr}. Expected YYYY-MM-DDTHH-MM-SSZ")


def format_for_api(dt: datetime.datetime) -> str:
    """
    NOAA API expects begin_date/end_date as YYYYMMDD or YYYYMMDD HH:mm.
    We'll use YYYYMMDD.
    """
    return dt.strftime("%Y%m%d")


# ---------------------------
# Worker
# ---------------------------

def run_download_noaa_obv_wl_from_params(
    *,
    output_dir: str,
    start_time: str,
    end_time: str,
    station_list: List[str],
    auto_find_if_empty: bool,
    station_discovery_type: str = "historicwl",
    station_discovery_base_url: str = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json",
    station_discovery_extra_params: Optional[Dict[str, Any]] = None,
    api_datagetter_base: str,
    api_datums_base_template: str,
    application: str,
    datum: str,
    units: str,
    time_zone: str,
    response_format: str,
    product_hourly: str,
    product_sixmin: str,
    use_sixmin: bool,
    extra_query_params: Optional[Dict[str, Any]] = None,
) -> None:
    if not os.path.exists(output_dir):
        raise RuntimeError(f"FATAL ERROR: output_dir '{output_dir}' does not exist!")

    # Parse times
    start_dt = parse_config_datetime(start_time)
    end_dt = parse_config_datetime(end_time)
    begin_date = format_for_api(start_dt)
    end_date = format_for_api(end_dt)

    # Stations
    sites: List[str] = list(station_list or [])
    if not sites and auto_find_if_empty:
        '''
        nc_path = os.path.join(output_dir, "sfincs_his.nc")
    
        if not os.path.exists(nc_path):
            raise RuntimeError(f"{nc_path} not found.")
        sites = get_waterlevel_station_ids()
        '''
        obs_path = os.path.join(output_dir, "sfincs.obs")

        if not os.path.exists(obs_path):
            raise RuntimeError(f"{obs_path} not found.")

        sites = get_waterlevel_station_ids_from_obs(obs_path)
        '''
        sites = find_waterlevel_stations(
            [],
            station_type=station_discovery_type,
            base_url=station_discovery_base_url,
            extra_params=station_discovery_extra_params,
        )
        '''
    if not sites:
        raise RuntimeError("No stations provided (station_list empty and auto discovery disabled/failed).")

    print(f"sites : {sites}")
    failed_sites: List[str] = []
    product = product_sixmin if use_sixmin else product_hourly
    api_datagetter_base = api_datagetter_base.rstrip("?")

    for sta in sites:
        try:
            params = {
                "product": product,
                "application": application,
                "begin_date": begin_date,
                "end_date": end_date,
                "datum": datum,
                "station": sta,
                "time_zone": time_zone,
                "units": units,
                "format": response_format,
            }
            if extra_query_params:
                params.update({k: v for k, v in extra_query_params.items() if v is not None})

            q = urllib.parse.urlencode(params)
            url = f"{api_datagetter_base}?{q}"
            print(f"[INFO] station={sta} → URL={url}")

            try:
                with urllib.request.urlopen(url) as rno:
                    json_data = json.loads(rno.read().decode("utf-8"))
            except Exception as e:
                print(f"[WARN] station {sta} skipped (data fetch failed): {e}")
                failed_sites.append(sta)
                continue

            out_json_path = os.path.join(output_dir, f"{sta}.json")
            with open(out_json_path, "w", encoding="utf-8") as jso:
                json.dump(json_data, jso, indent=2)

            datums_url = api_datums_base_template.format(station=sta)
            try:
                with urllib.request.urlopen(datums_url) as rno:
                    json_data = json.loads(rno.read().decode("utf-8"))
            except Exception as e:
                print(f"[WARN] station {sta} skipped (datums fetch failed): {e}")
                failed_sites.append(sta)
                continue

            out_datum_path = os.path.join(output_dir, f"{sta}_datum.json")
            with open(out_datum_path, "w", encoding="utf-8") as jso:
                json.dump(json_data, jso, indent=2)

            print(datetime.datetime.now(), "---", f"Downloaded data for station: {sta}")

        except Exception as e:
            print(f"[ERROR] station {sta} unexpected failure: {e}")
            failed_sites.append(sta)

    if failed_sites:
        print("[SUMMARY] Failed stations:", ", ".join(failed_sites))
    else:
        print("[SUMMARY] All stations downloaded successfully.")


# ---------------------------------------------
# Config reader
# ---------------------------------------------

def read_config_and_run(config_path: str) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    run_download_noaa_obv_wl_from_params(**cfg)


def main():
    parser = argparse.ArgumentParser(description="Download NOAA observed water level data using a YAML config.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    args = parser.parse_args()
    read_config_and_run(args.config)


if __name__ == "__main__":
    main()

