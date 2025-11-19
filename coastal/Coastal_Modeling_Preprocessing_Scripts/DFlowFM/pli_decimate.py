"""
PLI Decimate script.

Clip a PLI boundary to a desired polygon.

Usage:
python pli_decimate.py -c MyPolygon.txt -o Clipped.pli Boundary.pli
    Clips Boundary.pli to the points that lie inside MyPolygon
    and writes the result to Clipped.pli

python pli_decimate.py -c MyPolygon.txt -o Clipped.pli -n 10 Boundary.pli
    Clips Boundary.pli to the points that lie in inside MyPolygon.
    When the output is written, only every 10th point is written to Clipped.pli.
    Note that clipping happens before decimation.

python pli_decimate.py -n 10 -o Decimated.pli Boundary.pli
    Only do decimation of Boundary.pli by writing every 10th point to Decimated.pli
"""

import argparse
import pathlib
import numpy as np
from itertools import islice, compress
from common.io import read_pli, write_pli, read_polygon
from common.geometry import clip_point_to_roi

def process_stations(d, polygon=None, n=1):
    if polygon is not None:
        idxs = clip_point_to_roi(polygon, d['values'])
        d['values'] = d['values'][idxs]
        d['index'] = list(compress(d['index'], idxs))

    new_index, new_values = zip(*islice(zip(d['index'], d['values']), 0, None, n))
    d['index'] = list(new_index)
    d['values'] = np.asarray(new_values, dtype='float64')
    return d
        

def main(args):
    pli = read_pli(args.pli)
    
    if args.polygon:
        P = read_polygon(args.polygon)
    else:
        P = None
    
    pli = process_stations(pli, polygon=P, n=args.n)
    write_pli(args.output, pli)


def get_options():
    parser = argparse.ArgumentParser()

    parser.add_argument('pli', type=pathlib.Path, help="Path to PLI boundary file")
    parser.add_argument('-n', type=int, default=1, help="Decimation factor")
    parser.add_argument('-o', '--output', type=pathlib.Path, help="Output file path")
    parser.add_argument("-c", "--clip", dest="polygon", default=None, type=pathlib.Path, help="Use a polygon to select sites for output")
    args = parser.parse_args()

    if args.output is None:
        out = pathlib.Path()/args.pli.name
        if out.exists():
            raise RuntimeError("-o must be defined")
        else:
            args.output = pathlib.Path()/args.pli.name
    return args


if __name__ == "__main__":
    args = get_options()
    main(args)
