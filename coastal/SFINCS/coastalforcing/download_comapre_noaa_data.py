import argparse
import os
import sys
import subprocess
import yaml
from datetime import datetime
import shutil
import re
import traceback
from datetime import datetime, timezone
from pathlib import Path
from download_data.download_noaa_obv_wl_sfincs import get_waterlevel_station_ids_from_nc, get_waterlevel_station_ids_from_obs, run_download_noaa_obv_wl_from_params
from process_data.compare_sfincs_map_vs_noaa import compare_sfincs_his_vs_noaa

def run_sfincs_cmd(forcing_output_dir: str) -> None:
    """
    Run the SFINCS docker container with the given forcing_output_dir mounted to /data.
    Waits for completion and reports success or errors.
    """
    cmd = [
        "sudo", "docker", "run", "-ti",
        "-v", f"{os.path.abspath(forcing_output_dir)}:/data",
        "deltares/sfincs-cpu"
    ]

    try:
        print(f"[INFO] Running SFINCS with directory: {forcing_output_dir}")
        result = subprocess.run(
            cmd,
            check=True,            # raises CalledProcessError if non-zero exit
            capture_output=True,   # capture both stdout and stderr
            text=True
        )
        print("[SUCCESS] SFINCS completed successfully.")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("[ERROR] SFINCS run failed.")
        print("Exit code:", e.returncode)
        print("Stdout:", e.stdout)
        print("Stderr:", e.stderr)
        raise

def _parse_utc(s: str) -> str:
    """
    Accept common ISO-8601 strings:
      - 'YYYY-MM-DDTHH:MM:SSZ'
      - 'YYYY-MM-DDTHH:MM:SS+00:00'
      - 'YYYY-MM-DD HH:MM:SSZ'
      - 'YYYY-MM-DDTHH-MM-SSZ'  (your legacy format)
    Return normalized string 'YYYY-MM-DDTHH-MM-SSZ' to preserve your downstream expectations.
    """
    raw = s.strip()

    # If already in your legacy 'T%H-%M-%SZ' format, accept it
    try:
        dt = datetime.strptime(raw, "%Y-%m-%dT%H-%M-%SZ").replace(tzinfo=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H-%M-%SZ")
    except Exception:
        pass

    # Normalize common ISO variants to aware UTC
    s2 = raw.replace(" ", "T")
    if s2.endswith(("Z", "z")):
        s2 = s2[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s2).astimezone(timezone.utc)
        return dt.strftime("%Y-%m-%dT%H-%M-%SZ")
    except Exception as e:
        raise ValueError(f"Unrecognized UTC time format '{s}': {e}")

def _normpath(*parts) -> str:
    return str(Path(*parts).resolve())

def _must_exist(path: str, label: str):
    if not Path(path).exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Download NOAA observed WL and compare with SFINCS output"
    )

    parser.add_argument("--config", default=None, help="Path to YAML config file (optional)")

    parser.add_argument("--start_date", help="Start date (YYYY-MM-DDTHH-MM-SSZ)")
    parser.add_argument("--end_date", help="End date (YYYY-MM-DDTHH-MM-SSZ)")
    parser.add_argument("--sim_dir", help="Simulation directory")
    parser.add_argument("--noaa_output_dir", default=None, help="NOAA output directory")
    parser.add_argument("--run_sfincs", action="store_true",
                        help="Run SFINCS Docker after setup")

    args = parser.parse_args()

    params = {}

    # ---- priority 1: config file ----
    if args.config:
        if not os.path.isfile(args.config):
            print(f"[error] Config file not found: {args.config}")
            sys.exit(1)
        cfg = load_config(args.config)
        params["start_date"] = cfg.get("start_date")
        params["end_date"] = cfg.get("end_date")
        params["sim_dir"] = cfg.get("sim_dir")
        params["noaa_output_dir"] = cfg.get("noaa_output_dir")
        params["run_sfincs"] = bool(cfg.get("run_sfincs", False))

    # ---- priority 2: CLI arguments ----
    else:
        params["start_date"] = args.start_date
        params["end_date"] = args.end_date
        params["sim_dir"] = args.sim_dir
        params["noaa_output_dir"] = args.noaa_output_dir
        params["run_sfincs"] = bool(args.run_sfincs)

    # ---- check required params ----
    if not params["start_date"] or not params["end_date"] or not params["sim_dir"]:
        print(
            "\n[info] Missing required parameters.\n"
            "Please provide them either via:\n"
            "  1) A config YAML file:  --config config.yaml\n"
            "     with keys: start_date, end_date, sim_dir, noaa_output_dir (optional), run_sfincs (optional)\n"
            "  2) Or command line arguments:\n"
            "     --start_date YYYY-MM-DDTHH-MM-SSZ --end_date YYYY-MM-DDTHH-MM-SSZ --sim_dir <path>\n"
        )
        sys.exit(0)

    # ---- parse datetimes ----
    '''
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%dT%H-%M-%SZ")
        end_dt = datetime.strptime(end_date, "%Y-%m-%dT%H-%M-%SZ")
    except ValueError as e:
        raise ValueError(f"Invalid date format: {e}")

    print(f"[main] start: {start_dt}, end: {end_dt}")
    print(f"[main] simulation directory: {sim_dir}")
    '''

    # ---- ensure sim_dir exists ----
    sim_dir = params['sim_dir']
    if not os.path.isdir(sim_dir):
        raise RuntimeError(f"Simulation directory not found: {sim_dir}")

    # ---- run sfincs if requested ----
    if params["run_sfincs"]:
        cmd = [
            "sudo", "docker", "run", "-ti",
            "-v", f"{os.path.abspath(sim_dir)}:/data",
            "deltares/sfincs-cpu"
        ]
        print("[main] Running SFINCS Docker:")
        print("       " + " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
            print("[main] SFINCS run finished successfully.")
        except subprocess.CalledProcessError as e:
            print(f"[main] ERROR running SFINCS: {e}")
            raise

    noaa_output_dir = params["noaa_output_dir"]
    if not noaa_output_dir:
        noaa_output_dir = sim_dir
    print(f"[main] NOAA output directory: {noaa_output_dir}")

    # Download NOAA data
    his_nc_path = os.path.join(sim_dir, 'sfincs_his.nc')
    station_list=params.get('station_list', get_waterlevel_station_ids_from_nc(his_nc_path))

    '''
    try:

        run_noaa_stations_from_params(
            geojson_path=cfg["geojson_path"],
            utm_epsg=32614,
            start_time=cfg['start_time'],
            end_time=cfg['end_time'],
            interval_minutes=int(cfg.get("interval_minutes", 6)),
            output_dir=sim_dir,
            obs_filename=os.path.join(sim_dir, "sfincs.obs"),
            out_filename=cfg.get("out_filename", "noaa.out"),
            station_types=cfg['station_types'],
        )

    except Exception as e:
        print(f"Error downloading from NOAA : {str(e)}")
        traceback.print_exc()

    '''

    print("\nDownloading NOAA output")

    try:
        run_download_noaa_obv_wl_from_params(
            output_dir=noaa_output_dir,
            start_time=params['start_date'],
            end_time=params['end_date'],
            station_list=station_list, #[8772985,8773146,8773259,8773701,8773767],
            auto_find_if_empty=True,
            station_discovery_type=params.get("station_discovery_type", "water_level"),
            station_discovery_base_url="https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json",
            station_discovery_extra_params=None,
            api_datagetter_base="https://api.tidesandcurrents.noaa.gov/api/prod/datagetter",
            api_datums_base_template="https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/{station}/datums.json?units=metric",
            application=params.get('application', "NOS.COOPS.TAC.WL"),
            datum=params.get('datum', 'MLLW'),
            units=params.get('units', 'metric'),
            time_zone=params.get('time_zone', 'GMT'),
            response_format=params.get('resplonse_format', 'json'),
            product_hourly=params.get('product_hourly', "hourly_height"),
            product_sixmin=params.get("product_sixmin", "water_level"),
            use_sixmin=True,
            extra_query_params=params.get('extra_query_params', {"interval": "6"})
        )

    except Exception as e:
        print(f"Error downloading from NOAA : {str(e)}")
        traceback.print_exc()

    print("\nCompairing SFINCS output with NOAA output")
    stations=list(map(str, station_list))
    print(stations)

    try:
        compare_sfincs_his_vs_noaa(
            sfincs_his_nc=his_nc_path,
            noaa_dir=noaa_output_dir,
            stations=stations,
            outdir=noaa_output_dir,
            datum_shift=0.0,
            resample_to="model",
            fig_dpi=140,
            verbose=True,
        )
    except Exception as e:
        print(f"Compairing SFINCS output with NOAA output : {str(e)}")
        traceback.print_exc()


def load_config(path: str):
    """Load YAML config file."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(str(e))
        traceback.print_exc()
