"""
EXT slice

Extract region of interest subset from boundary csv and
optionally slice EXT and streamlines to same region of interest.

Usage:
python ext_slice.py -o output_dir/ boundary.csv MyPolygon.txt
    Clip boundary.csv to the region of interest described by MyPolygon.txt
"""

import argparse
import pathlib
import numpy as np
import csv
import itertools

from common.geometry import clip_point_to_roi
from common.io import read_polygon, read_csv, read_ext, write_ext_v2


def create_csv_subdomain(src_csv, dst_csv, mask):
    """Read a source csv and write masked output to dst_csv.

    Reading the source from disk is necessary because NumPy will lowercase
    all column names
    """
    with open(dst_csv, 'w', newline='') as fout:
        csvw = csv.writer(fout)
        with open(src_csv, 'r') as fin:
            csvr = csv.reader(fin)
            csvw.writerow(next(csvr))
            for row in itertools.compress(csvr, mask):
                csvw.writerow(row)


# user defined options
def get_options():
    parser = argparse.ArgumentParser(description='Create ext DFlow subdomain file based on user specified Polygon.')
    parser.add_argument('boundary_csv', type=pathlib.Path,
                    help='CSV file mapping boundary ids to geospatial position.')
    parser.add_argument('polygon', type=pathlib.Path,
                    help='The path of the polygon file defining the region of interest.')
    parser.add_argument('-o', '--output', dest='output_dir', default=pathlib.Path('.'), type=pathlib.Path,
                    help='The directory to write DFlow subdomain ext file to')
    args = parser.parse_args()

    # Validate that output_dir is an existing directory
    if not args.output_dir.is_dir():
        raise NotADirectoryError(args.output_dir)

    return args

def main(args):

    # read Boundary ID dataframe containing geospatial information
    bnd_id_data = read_csv(args.boundary_csv)
    # extract polygon coordinate info from user defined file
    polygon_coords = read_polygon(args.polygon)

    # Extract boundary point coordinates
    bnd_pts = np.column_stack([bnd_id_data["long"], bnd_id_data["lat"]]).astype(float)
    poly_mask = np.array(clip_point_to_roi(polygon_coords, bnd_pts), dtype='bool')

    # Create NWM common ID boundary csv slice for land boundary discharge referencing
    csv_name = f"{args.boundary_csv.stem}_slice_{args.polygon.stem}{args.boundary_csv.suffix}"
    create_csv_subdomain(args.boundary_csv, args.output_dir.joinpath(csv_name), poly_mask)


##### User command line example  ########
#./python3.7 ../../DFlow_polygon_slice_ext.py --bnd_ids_csv="./InflowBoundaryConditions.csv" --polygon_file="./Irma_Enclosure.pol" --streamlines="./NWM_slice.csv" --output="./"


# Run main when this file is run
if __name__ == "__main__":
    args = get_options()
    main(args)

