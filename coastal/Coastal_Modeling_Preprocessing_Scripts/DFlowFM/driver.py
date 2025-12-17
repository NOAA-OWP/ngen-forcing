"""
Automation script for workflow.

All arguments are contained within a JSON file. The general workflow is

pli_decimate / ext_slice -> waterlevel / streamflow / lateral discharge

First the domain is clipped to the region of interest using pli_decimate and ext_slice.
Then waterlevel/streamflow/qlateral extraction is run on the outputs clipped to ROI.

The keys in the JSON file that need to be defined are:
start_time: Start time of simulation (YYYY-MM-DD_HH:MM:SS)
stop_time: Stop time of the simulation (YYYY-MM_DD_HH:MM:SS)
global_pli: PLI input
global_boundary: DFlow EXT boundary
region: Polygon file (1 vertex per line)
boundary_csv: Inflow boundary conditions
streamlines: Georeferencing of streamlines
fort63: ADCIRC output (netcdf format)
streamflow_input: Directory for streamflow data files

"""

import argparse
import json
import pathlib
import itertools
import random
import string
import datetime
import os
import pli_decimate
import xyn_decimate
import ext_slice
import waterlevel
import waterlevel_CERA
import waterlevel_extended
import waterlevel_SCHISM
import waterlevel_Tidal
#import waterlevel_STOFS_reanalysis
import lateral_discharge_HUC_agg_dflowfm
import NWM_polygon_Retro_NWM_Inflow
import NWM_polygon_TRoute_Inflow
import NWM_polygon_TRoute_Inflow_Great_Lakes
from common.io import read_ext
from common.io import init_ext_bnd, ext_bnd_condition, ext_bnd_discharge
import waterlevel_FVCOM_Lake_Erie
import waterlevel_FVCOM_Lake_Ontario
import waterlevel_FVCOM_Lake_Superior
import waterlevel_FVCOM_Lake_Michigan_Huron

FILE_KEYS = (
        "global_pli",
        "global_xyn",
        "region",
        "region_US",
        "region_Can",
        "fort63",
        "fort64",
        "CERA",
        "CERA2",
        "CERA1",
        "hgrid3",
        "source",
        "elev2d",
        "stofs",
        "map",
        "dflowfm_elev2d",
        "boundary_csv",
        "boundary_csv_US",
        "boundary_csv_Can",
        "HUC12_shpfile",
        "NWM_shpfile",
        "NWM_shpfile_US",
        "NWM_shpfile_Can",
        "HUC_to_COMMID_csvfile",
        "NWM_latq_csv"
)

DIR_KEYS = ("output_directory", "Forecast_streamflow","streamflow_input", "TRoute", "TRoute_US", "TRoute_Can", "FVCOM", "STOFS_Dir", "Atmospheric_Forcings")

CRITERIA_KEYS = ["discharge_max_filter", "atmospheric_model", "lake", "tidal"]

def check_keys(namespace, keys, raise_error=False):
    keys = set(keys)
    chk = keys.issubset(namespace.keys())
    if raise_error:
        raise KeyError(f"Missing {list(keys - namespace.keys())}")
    else:
        if(chk == "False"):
            return False
        else:
            return chk

def get_options():
    parser = argparse.ArgumentParser()

    parser.add_argument('config', type=pathlib.Path, help='Configuration JSON')

    args = parser.parse_args()
    return args

def validate_directory(path):
    """Ensure that path is a directory"""
    if not path.is_dir():
        raise NotADirectoryError(path)

def validate_file(path):
    """Ensure that path to a file exists"""
    if not path.is_file():
        raise FileNotFoundError(path)

def main(args):
    with open(args.config, 'r') as fh:
        config = json.load(fh)

    #parse dates
    start_time = datetime.datetime.strptime(config["start_time"], '%Y-%m-%d_%H:%M')
    stop_time = datetime.datetime.strptime(config["stop_time"], '%Y-%m-%d_%H:%M')

    try:
        schism_start_time = datetime.datetime.strptime(config["schism_start_time"], '%Y-%m-%d_%H:%M')
    except:
        print("No SCHISM start time found for waterlevels, skipping method to produced DFlow-FM waterlevels from SCHISM output.")

    for p in FILE_KEYS:
        try:
            config[p] = pathlib.Path(config[p])
            validate_file(config[p])
        except:
            print(str(p) + '  has not been specified in config file, skipping method.')
    for p in DIR_KEYS:
        try:
            config[p] = pathlib.Path(config[p])
            validate_directory(config[p])
        except:
            print(str(p) + '  has not been specified in config file, skipping method.')

    for p in CRITERIA_KEYS:
        if(p == "discharge_max_filter"):
            try:
                float(config[p])
                config[p] = float(config[p])
            except:
                print(str(p) + '  has not been specified in config file or is an improper digit value, skipping discharge critera method.')
        elif(p == "atmospheric_model"):
            try:
                if(str(config[p]).upper() == "AORC" or str(config[p]).upper() == "NWM"):
                    config[p] = str(config[p])
            except:
                print(str(p) + '  has not been specified or is not a supported atmospheric model currently, skipping D-Flow FM atmospheric forcing file production method.')

        elif(p == "lake"):
            try:
                if(str(config[p]).upper() == "ONTARIO" or str(config[p]).upper() == "ERIE" or str(config[p]).upper() == "SUPERIOR" or str(config[p]).upper() == "MICHIGAN-HURON"):
                    config[p] = str(config[p])
            except:
                print(str(p) + '  has not been specified or is not a supported great lake option (Erie, Ontario, Superior, Michigan-Huron) currently, skipping FVCOM water level extraction method.')

    try:
        region_stem = config["region"].stem
    except: 
        region_stem = config["region_US"].stem

    out_dir = config["output_directory"]


    # Create a discharge filter configuration option flag
    # that will create discharge option for moudle to ingest
    try:
        discharge_max_filter = float(config["discharge_max_filter"])
    except:
        discharge_max_filter = 0.0

    # Initalize ext bnd file format
    ext_out = out_dir.joinpath(f"FlowFM_bnd_{region_stem}.ext")
    init_ext_bnd(ext_out)

    # Check if global_pli and region are defined then we clip pli
    if check_keys(config, ("global_pli", "region")):
        print("PLI Decimate...")
        pli_decimate_args = argparse.Namespace()
        pli_decimate_args.n = 1
        pli_decimate_args.pli = config["global_pli"]
        pli_decimate_args.polygon = config["region"]
        pli_out = out_dir.joinpath(f"{pli_decimate_args.pli.stem}.pli")
        pli_decimate_args.output = pli_out
        print(pli_decimate_args)
        pli_decimate.main(pli_decimate_args)
        assert pli_out.exists()
        print(pli_out)

    if check_keys(config, ("global_pli", "region_US")):
        print("PLI Decimate...")
        pli_decimate_args = argparse.Namespace()
        pli_decimate_args.n = 1
        pli_decimate_args.pli = config["global_pli"]
        pli_decimate_args.polygon = config["region_US"]
        pli_out = out_dir.joinpath(f"{pli_decimate_args.pli.stem}.pli")
        pli_decimate_args.output = pli_out
        print(pli_decimate_args)
        pli_decimate.main(pli_decimate_args)
        assert pli_out.exists()
        print(pli_out)

    if check_keys(config, ("global_xyn", "region")):
        print("XYN Decimate...")
        xyn_decimate_args = argparse.Namespace()
        xyn_decimate_args.n = 1
        xyn_decimate_args.xyn = config["global_xyn"]
        xyn_decimate_args.polygon = config["region"]
        xyn_out = out_dir.joinpath(f"{xyn_decimate_args.xyn.stem}_slice_{region_stem}.xyn")
        xyn_decimate_args.output = xyn_out
        print(xyn_decimate_args)
        xyn_decimate.main(xyn_decimate_args)
        assert xyn_out.exists()
        print(xyn_out)

    if check_keys(config, ("region", "boundary_csv")):
        print("NWM csv Slice...")
        ext_slice_args = argparse.Namespace()
        ext_slice_args.boundary_csv = config["boundary_csv"]
        ext_slice_args.polygon = config["region"]
        ext_slice_args.output_dir = out_dir
        ext_slice.main(ext_slice_args)
        [boundary_csv] = list(out_dir.glob(f"{config['boundary_csv'].stem}_slice_{region_stem}.csv"))
        print(boundary_csv)

    if check_keys(config, ("region_US", "boundary_csv_US")):
        print("NWM US csv Slice...")
        ext_slice_args = argparse.Namespace()
        ext_slice_args.boundary_csv = config["boundary_csv_US"]
        ext_slice_args.polygon = config["region_US"]
        ext_slice_args.output_dir = out_dir
        ext_slice.main(ext_slice_args)
        [boundary_csv_US] =  list(out_dir.glob(f"{config['boundary_csv_US'].stem}_slice_{region_stem}.csv"))
        print(boundary_csv_US)

    if check_keys(config, ("region_Can", "boundary_csv_Can")):
        print("NWM Can csv Slice...")
        ext_slice_args = argparse.Namespace()
        ext_slice_args.boundary_csv = config["boundary_csv_Can"]
        ext_slice_args.polygon = config["region_Can"]
        ext_slice_args.output_dir = out_dir
        ext_slice.main(ext_slice_args)
        [boundary_csv_Can] = list(out_dir.glob(f"{config['boundary_csv_Can'].stem}_slice_{region_stem}.csv"))
        print(boundary_csv_Can)

    if check_keys(config, ("global_pli", "map")):
        print("Extracting DFlowFM waterlevels for TRoute Input...")
        DFlowFM_TRoute_Data_args = argparse.Namespace()
        DFlowFM_TRoute_Data_args.pli = config["global_pli"]
        DFlowFM_TRoute_Data_args.map = config["map"]
        DFlowFM_TRoute_Data_out = out_dir.joinpath(f"DFlowFM_Waterlevels_TRoute_{region_stem}.nc")
        DFlowFM_TRoute_Data_args.output = DFlowFM_TRoute_Data_out
        print(DFlowFM_TRoute_Data_args)
        DFlowFM_waterlevel_to_TRoute.main(DFlowFM_TRoute_Data_args)
        assert DFlowFM_TRoute_Data_out.exists()
        print(DFlowFM_TRoute_Data_out)

    if check_keys(config, ("tidal","global_pli")):
        print("Tidal Waterlevel extraction...")
        Tidal_waterlevel_args = argparse.Namespace()
        Tidal_waterlevel_args.pli = pli_out
        Tidal_waterlevel_args.start_time = start_time
        Tidal_waterlevel_args.stop_time = stop_time
        wl_out = out_dir.joinpath(f"Tidal_waterlevel_slice_{region_stem}.bc")
        Tidal_waterlevel_args.output = wl_out
        waterlevel_Tidal.main(Tidal_waterlevel_args)
        assert wl_out.exists()
        print(wl_out)

        ext_bnd_condition(ext_out, "waterlevelbnd", pli_out, wl_out)

    if check_keys(config, ("fort63","global_pli")):
        print("ADCIRC Waterlevel extraction...")
        waterlevel_args = argparse.Namespace()
        waterlevel_args.fort63 = config["fort63"]
        waterlevel_args.pli = pli_out
        #waterlevel_args.boundary_csv = boundary_csv
        wl_out = out_dir.joinpath(f"waterlevel_slice_{region_stem}.bc")
        waterlevel_args.output = wl_out
        waterlevel.main(waterlevel_args)
        assert wl_out.exists()
        print(wl_out)

        ext_bnd_condition(ext_out, "waterlevelbnd", pli_out, wl_out)


    if check_keys(config, ("CERA","global_pli")):
        print("CERA Waterlevel extraction...")
        CERA_waterlevel_args = argparse.Namespace()
        CERA_waterlevel_args.CERA = config["CERA"]
        CERA_waterlevel_args.start_time = start_time
        CERA_waterlevel_args.stop_time = stop_time
        CERA_waterlevel_args.pli = pli_out
        wl_out = out_dir.joinpath(f"CERA_waterlevel_slice_{region_stem}.bc")
        CERA_waterlevel_args.output = wl_out
        waterlevel_CERA.main(CERA_waterlevel_args)
        assert wl_out.exists()
        print(wl_out)

        ext_bnd_condition(ext_out, "waterlevelbnd", pli_out, wl_out)

    if check_keys(config, ("CERA1","CERA2","global_pli")):
        print("CERA Waterlevel extended extraction...")
        waterlevel_extended_args = argparse.Namespace()
        waterlevel_extended_args.CERA1 = config["CERA1"]
        waterlevel_extended_args.CERA2 = config["CERA2"]
        waterlevel_extended_args.start_time = start_time
        waterlevel_extended_args.stop_time = stop_time
        waterlevel_extended_args.pli = pli_out
        wl_out = out_dir.joinpath(f"CERA_waterlevel_extended_slice_{region_stem}.bc")
        waterlevel_extended_args.output = wl_out
        waterlevel_extended.main(waterlevel_extended_args)
        assert wl_out.exists()
        print(wl_out)

        ext_bnd_condition(ext_out, "waterlevelbnd", pli_out, wl_out)

    if check_keys(config, ("STOFS_Dir","global_pli","start_time", "stop_time")):
        print("STOFS Reanalysis Waterlevel extraction...")
        waterlevel_STOFS_args = argparse.Namespace()
        waterlevel_STOFS_args.STOFS = config["STOFS_Dir"]
        waterlevel_STOFS_args.pli = pli_out
        waterlevel_STOFS_args.start_time = start_time
        waterlevel_STOFS_args.stop_time = stop_time
        wl_out = out_dir.joinpath(f"STOFS_Reanalysis_Waterlevel_slice_{region_stem}.bc")
        waterlevel_STOFS_args.output = wl_out
        waterlevel_STOFS_reanalysis.main(waterlevel_STOFS_args)
        assert wl_out.exists()
        print(wl_out)

        ext_bnd_condition(ext_out, "waterlevelbnd", pli_out, wl_out)

    if check_keys(config, ("global_pli", "hgrid3", "elev2d","schism_start_time")):
        print("SCHISM Waterlevel extraction...")
        waterlevel_SCHISM_args = argparse.Namespace()
        waterlevel_SCHISM_args.elev2d = config["elev2d"]
        #waterlevel_SCHISM_args.dflowfm_elev2d = config["dflowfm_elev2d"]
        waterlevel_SCHISM_args.hgrid3 = config["hgrid3"]
        waterlevel_SCHISM_args.pli = pli_out
        waterlevel_SCHISM_args.schism_start_time = schism_start_time
        wl_out = out_dir.joinpath(f"DFLOWFM_waterlevel_slice_{region_stem}_from_SCHISM_elev2d.bc")
        waterlevel_SCHISM_args.output = wl_out
        waterlevel_SCHISM_args.output_dir = out_dir
        waterlevel_SCHISM.main(waterlevel_SCHISM_args)
        assert wl_out.exists()
        print(wl_out)

        ext_bnd_condition(ext_out, "waterlevelbnd", pli_out, wl_out)

    if check_keys(config, ("FVCOM","global_pli","start_time", "stop_time")):
        print("FVCOM Waterlevel extraction...")
        waterlevel_FVCOM_args = argparse.Namespace()
        waterlevel_FVCOM_args.FVCOM = config["FVCOM"]
        waterlevel_FVCOM_args.Lake = config["lake"]
        waterlevel_FVCOM_args.pli = pli_out
        waterlevel_FVCOM_args.start_time = start_time
        waterlevel_FVCOM_args.stop_time = stop_time
        wl_out = out_dir.joinpath(f"FVCOM_waterlevel_slice_{region_stem}.bc")
        waterlevel_FVCOM_args.output = wl_out
        if(str(config["lake"]).upper() == "ERIE"):
            waterlevel_FVCOM_Lake_Erie.main(waterlevel_FVCOM_args)
        elif(str(config["lake"]).upper() == "ONTARIO"):
            waterlevel_FVCOM_Lake_Ontario.main(waterlevel_FVCOM_args)
        elif(str(config["lake"]).upper() == "MICHIGAN_HURON"):
            waterlevel_FVCOM_Lake_Michigan_Huron.main(waterlevel_FVCOM_args)
        elif(str(config["lake"]).upper() == "SUPERIOR"):
            waterlevel_FVCOM_Lake_Superior.main(waterlevel_FVCOM_args)
        assert wl_out.exists()
        print(wl_out)

        ext_bnd_condition(ext_out, "waterlevelbnd", pli_out, wl_out)

    # Flag to indicate if there is any HUC-12 or NWM polygon shapefiles available
    # from user inputs
    if (check_keys(config, ("streamflow_input",  "start_time", "stop_time", "region"))):
        qlateral_args = argparse.Namespace()
        if (check_keys(config, ("HUC12_shpfile","HUC_to_COMMID_csvfile"))):
            print("Lateral flow HUC-12 aggregation method for DFlowFM extraction...")
            qlateral_args.HUC_agg_csv = config['HUC_to_COMMID_csvfile']
            qlateral_args.HUC_polygon = config['HUC12_shpfile']
            qlateral_args.NWC_polygon = None
            qlateral_args.polygon = config["region"]
            qlateral_args.start_time = start_time
            qlateral_args.stop_time = stop_time
            qlateral_args.input_dir = config["streamflow_input"]
            qlateral_args.ext = ext_out
            ql_out = out_dir.joinpath(f"qlateral_HUC12_slice_{region_stem}.bc")
            qlateral_args.output = ql_out
            print(qlateral_args)
            lateral_discharge_HUC_agg_dflowfm.main(qlateral_args)
            assert ql_out.exists()
            print(ql_out)
        else:
            print("No polygon shapefile found to create area for lateral discharges, exiting method and not including lateral discharges in production.")


    if (check_keys(config, ("streamflow_input",  "start_time", "stop_time", "region","NWM_shpfile"))):
        print("Extracting NWM polygon inflows from NWM retrospective data...")
        inflow_retro_args = argparse.Namespace()
        inflow_retro_args.comm_ids = boundary_csv
        inflow_retro_args.NWM_polygon = config['NWM_shpfile']
        inflow_retro_args.polygon = config["region"]
        inflow_retro_args.start_time = start_time
        inflow_retro_args.stop_time = stop_time
        inflow_retro_args.input_dir = config["streamflow_input"]
        inflow_retro_args.ext = ext_out
        inflow_out = out_dir.joinpath(f"NWM_polygon_retro_inflow_slice_{region_stem}.bc")
        inflow_retro_args.output = inflow_out
        print(inflow_retro_args)
        NWM_polygon_Retro_NWM_Inflow.main(inflow_retro_args)
        assert inflow_out.exists()
        print(inflow_out)

    if (check_keys(config, ("TRoute",  "start_time", "stop_time", "region","NWM_shpfile")) and os.path.isfile(boundary_csv)):
        print("Extracting NWM polygon inflows from TRoute data...")
        TRoute_inflow_args = argparse.Namespace()
        TRoute_inflow_args.comm_ids = boundary_csv
        TRoute_inflow_args.NWM_polygon = config['NWM_shpfile']
        TRoute_inflow_args.polygon = config["region"]
        TRoute_inflow_args.start_time = start_time
        TRoute_inflow_args.stop_time = stop_time
        TRoute_inflow_args.troute_input = config["TRoute"]
        TRoute_inflow_args.ext = ext_out
        TRoute_inflow_out = out_dir.joinpath(f"NWM_polygon_TRoute_inflow_slice_{region_stem}.bc")
        TRoute_inflow_args.output = TRoute_inflow_out
        print(TRoute_inflow_args)
        NWM_polygon_TRoute_Inflow.main(TRoute_inflow_args)
        assert TRoute_inflow_out.exists()
        print(TRoute_inflow_out)

    if (check_keys(config, ("TRoute_US", "TRoute_Can",  "start_time", "stop_time", "region_US", "region_Can", "NWM_shpfile_US", "NWM_shpfile_Can")) and os.path.isfile(boundary_csv_US) and os.path.isfile(boundary_csv_Can)):
        print("Extracting NWM polygon inflows from TRoute data...")
        TRoute_inflow_GL_args = argparse.Namespace()
        TRoute_inflow_GL_args.comm_ids_US = boundary_csv_US
        TRoute_inflow_GL_args.comm_ids_Can = boundary_csv_Can
        TRoute_inflow_GL_args.NWM_polygon_US = config['NWM_shpfile_US']
        TRoute_inflow_GL_args.NWM_polygon_Can = config['NWM_shpfile_Can']
        TRoute_inflow_GL_args.polygon_US = config["region_US"]
        TRoute_inflow_GL_args.polygon_Can = config["region_Can"]
        TRoute_inflow_GL_args.start_time = start_time
        TRoute_inflow_GL_args.stop_time = stop_time
        TRoute_inflow_GL_args.troute_input_US = config["TRoute_US"]
        TRoute_inflow_GL_args.troute_input_Can = config["TRoute_Can"]
        TRoute_inflow_GL_args.ext = ext_out
        TRoute_inflow_out_GL = out_dir.joinpath(f"NWM_polygon_TRoute_inflow_slice_{region_stem}.bc")
        TRoute_inflow_GL_args.output = TRoute_inflow_out_GL
        print(TRoute_inflow_GL_args)
        NWM_polygon_TRoute_Inflow_Great_Lakes.main(TRoute_inflow_GL_args)
        assert TRoute_inflow_out_GL.exists()
        print(TRoute_inflow_out_GL)

    #if (check_keys(config, ("start_time","stop_time","NWM_latq_csv","streamflow_input"))):
    #    print("NWM source point lateral discharge extraction...")
    #    source_NWM_args = argparse.Namespace()
    #    source_NWM_args.NWM_latq_csv = config["NWM_latq_csv"]
    #    source_NWM_args.input_dir = config["streamflow_input"]
    #    source_NWM_args.start_time = start_time
    #    source_NWM_args.stop_time = stop_time
    #    source_NWM_args.ext = ext_out
    #    ql_out = out_dir.joinpath(f"qlateral_NWM_Point_Source_{region_stem}.bc")
    #    source_NWM_args.output = ql_out
    #    NWM_source_to_DFlow.main(source_NWM_args)

    #if check_keys(config, ("Atmospheric_Forcings")):
    #    print("Creating DFlowFM Atmospheric Forcings File...")
    #    _args = argparse.Namespace()
    #    dflowfm_clip_forcings_args.forcings = config["Atmospheric_Forcings"]
    #    dflowfm_clip_forcings_args.polygon = config["region"]
    #    forcings_out = out_dir.joinpath(f"DFlowFM_Atmospheric_Forcings_{region_stem}.nc")
    #    dflowfm_clip_forcings_args.output = forcings_out
    #    dflowfm_clip_forcings.main(dflowfm_clip_forcings_args)
    #    assert forcings_out.exists()
    #    print(forcings_out)

if __name__ == "__main__":
    args = get_options()
    main(args)


