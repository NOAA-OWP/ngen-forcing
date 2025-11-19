import pandas as pd
import numpy as np
import argparse
import pathlib
from sklearn.neighbors import BallTree


def buffer_to_dict(path, return_boundaries=True):
    from collections import defaultdict
    def first_el(line):
        return line.split(None, 1)[0].strip()

    buf = open(path, 'r')
    description = next(buf)
    rv = {'description': description}
    NE, NP = map(int, next(buf).split())
    nodes = np.loadtxt(buf, max_rows=NP)
    columns = ['x', 'y', 'z'][:nodes.shape[1] - 1]
    rv['nodes'] = pd.DataFrame(nodes[:, 1:], index=pd.Index(nodes[:,0], dtype=int), columns=columns)
    rv['nodes'] = rv['nodes'].sort_index()

    elements = np.loadtxt(buf, dtype=int, max_rows=NE)
    rv['elements'] = pd.DataFrame(elements[:, 1:], index=pd.Index(elements[:, 0], dtype=int))
    rv['elements'] = rv['elements'].sort_index()

    if not return_boundaries:
        return rv
   # Assume EOF if NOPE is empty.
    try:
        NOPE = int(next(buf).split()[0])
    except IndexError:
        return rv
    # let NOPE=-1 mean an ellipsoidal-mesh
    # reassigning NOPE to 0 until further implementation is applied.
    boundaries = defaultdict(dict)
    _bnd_id = 0
    next(buf)
    while _bnd_id < NOPE:
        NETA = int(next(buf).split()[0])
        _cnt = 0
        boundaries[None][_bnd_id] = ind = []
        while _cnt < NETA:
            ind.append(int(first_el(next(buf))))
            _cnt += 1
        _bnd_id += 1
    NBOU = int(next(buf).split()[0])
    _nbnd_cnt = 0
    buf.readline()
    while _nbnd_cnt < NBOU:
        npts, ibtype = map(int, next(buf).split()[:2])
        _pnt_cnt = 0
        if ibtype not in boundaries:
            _bnd_id = 0
        else:
            _bnd_id = len(boundaries[ibtype])
        boundaries[ibtype][_bnd_id] = ind = []
        while _pnt_cnt < npts:
            line = next(buf).split()
            if len(line) == 1:
                ind.append(int(line[0]))
            else:
                index_construct = []
                for val in line:
                    if '.' in val:
                        continue
                    index_construct.append(val)
                ind.append(list(map(int, index_construct)))
            _pnt_cnt += 1
        _nbnd_cnt += 1
    buf.close()
    rv['boundaries'] = dict(boundaries)
    return rv
    

def nodes_to_coords(mesh, nodes):
    """Get coordinates of a set of nodes"""
    coords = []
    _nodetable = mesh['nodes']
    return _nodetable.loc[nodes, ['y', 'x']]
    
    
def get_options():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('grid', type=pathlib.Path)
    parser.add_argument('csv', type=pathlib.Path)
    parser.add_argument('--out-prefix', default='', type=str, help='Output file prefix')
    
    args = parser.parse_args()
    return args
    
    
def main(args):
    print("Reading csv...")
    pts = pd.read_csv(args.csv)
    pts['Poi'] = np.array(pts['id'],dtype=int)
    print("Reading grid...")
    mesh = buffer_to_dict(args.grid)
    
    boundary = set()
    for _, b in mesh['boundaries'].items():
        for _, b1 in b.items():
            boundary.update(b1)
            
    print("Finding closest elements...")
    # Boundary  node ids
    boundary = np.fromiter(boundary, dtype='int64')
    ipts = pts[['lat', 'long']].to_numpy()
    bpts = nodes_to_coords(mesh, boundary)
    # Haversine distance assumes lat, long coordinate order.
    tree = BallTree(bpts, metric='haversine')
    
    # idxs has the indexes into bpts/boundary
    dist, idxs = tree.query(ipts, k=2)
    # element_nodes are the relevant boundary nodes (2 closest nodes for each ipt)
    element_nodes = boundary[idxs]
    elements = mesh['elements'].values[:, 1:]
    # Find all elements that have a node queries from balltree
    elements_mask = np.isin(elements, element_nodes)
    # m contains all the elements that have two nodes (the 2 closest nodes)
    m = np.count_nonzero(elements_mask, axis=1) == 2
    elements = elements[m]
    elements_ids = mesh['elements'].index[m]
    
    print("Mapping ids to elements...")
    rv = {}
    nodes = {}
    for i, _id in enumerate(pts.hl_link):
        enodes = element_nodes[i]
        elidx = np.isin(elements, enodes).sum(axis=1).argmax()
        nwmel = elements_ids[elidx]
        rv[_id] = nwmel
        nodes[_id] = elements[elidx]
       
    rv = pd.DataFrame({'Element': rv, 'Node': nodes}, index=pd.Index(rv.keys(), name='Id'))
    pts = pts.set_index('hl_link')
    rv['Poi'] = pts['Poi']
    rv = rv.explode("Node")
    coords = mesh['nodes'].loc[rv.Node.unique(), ["x", "y"]].rename(columns={"x": "Longitude", "y": "Latitude"})
    rv = rv.merge(coords, left_on="Node", how="left", right_index=True)
    m = pts.start_DomL != pts.end_DomLoc
    rv = rv.loc[m]
    rv.reset_index().set_index("Node").to_csv(f"{args.out_prefix}element_mapping.csv")
    
    with open(f"{args.out_prefix}source_sink.in", 'w') as out:
        srcs = (pts.start_DomL == "o") & (pts.end_DomLoc == "i")
        sinks = (pts.start_DomL == "i") & (pts.end_DomLoc == "o")
        
        for m in (srcs, sinks):            
            out.write(f"{np.count_nonzero(m)}\n")
            els = rv.loc[m, "Element"]
            nids = pts.loc[m, "Poi"]
            for e, i in zip(els.values, nids.values):
                out.write(f"{e}\n")
            out.write("\n")

    with open(f"{args.out_prefix}source_sink_BMI.in", 'w') as out:
        srcs = (pts.start_DomL == "o") & (pts.end_DomLoc == "i")
        sinks = (pts.start_DomL == "i") & (pts.end_DomLoc == "o")

        for m in (srcs, sinks):
            out.write(f"{np.count_nonzero(m)}\n")
            els = rv.loc[m, "Element"]
            nids = pts.loc[m, "Poi"]
            for e, i in zip(els.values, nids.values):
                out.write(f"{e} {i}\n")
            out.write("\n")


if __name__ == "__main__":
    args = get_options()
    main(args)

