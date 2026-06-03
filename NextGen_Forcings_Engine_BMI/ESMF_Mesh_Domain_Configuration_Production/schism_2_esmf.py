import argparse
import netCDF4
import numpy as np

"""
Script to perform a conversion between an ASCII (horizontal grid) gr3 file and ESMF Unstructured Grid Format

Example Usage:  ./schism_2_esmf.py hgrid.gr3 SCHISM_ESMF_Mesh.nc --filter_open_bnds=True
"""


class gr3:
    def __init__(self,filename,num_elem,num_nodes):
        self.filename = filename
        self.num_elem = int(num_elem)
        self.num_nodes = int(num_nodes)
        self.lons = [-9999.0]*self.num_nodes
        self.lats = [-9999.0]*self.num_nodes
        self.elevs = [-9999.0]*self.num_nodes
        self.elems = [[]]*self.num_elem
        self.elem_elevs = [[]]*self.num_elem
        self.elem_coords = [[]]*self.num_elem
    def max_elem_len(self):
        elem_lens = self.elem_lens()
        return max(elem_lens) if elem_lens else 0
    def elem_lens(self):
        return [len(elem) for elem in self.elems]
    def padded_elems(self,pad_val=-1):
        max_elem_len = self.max_elem_len()
        return [elem + [pad_val]*(max_elem_len-len(elem)) for elem in self.elems]
    def element_coordinate(self, index):
        nodes = self.elems[index]
        lons = map(self.lons.__getitem__, nodes)
        lats = map(self.lats.__getitem__, nodes)
        center_lon = sum(lons) / 3
        center_lat = sum(lats) / 3
        return [center_lon, center_lat]
    def element_elevations(self, index):
        nodes = self.elems[index]
        elem_elevs = map(self.elevs.__getitem__, nodes)
        center_elev = sum(elem_elevs) / 3
        return center_elev
    def __str__(self):
        return "filename: %s\nnum_elements: %d\nnum_nodes: %d\nlons: \n%s\nlats: \n%s\nelevs: \n%s\nelem_nodes:\n%s" % (self.filename, self.num_elem, self.num_nodes, ",\n".join("%.4f" % lon for lon in self.lons),",\n".join("%.4f" % lat for lat in self.lats),",\n".join("%.4f" % elev for elev in self.elevs),",\n".join("%s" % elem for elem in self.elems))

def filter_nodes(gr3_obj, node_list):
    filt_gr3_obj = gr3(gr3_obj.filename, 0, 0)
    node_set = set(node_list)
    filt_gr3_obj.elems = [elem for elem in gr3_obj.elems if any([node in node_set for node in elem])]
    elem_node_set = set([node for elem in filt_gr3_obj.elems for node in elem])
    filt_idxs = [i for i in range(gr3_obj.num_nodes) if i in elem_node_set]
    filt_mapping = {idx: i for i, idx in enumerate(filt_idxs)}
    filt_gr3_obj.lons = [gr3_obj.lons[idx] for idx in filt_idxs]
    filt_gr3_obj.lats = [gr3_obj.lats[idx] for idx in filt_idxs]
    filt_gr3_obj.elevs = [gr3_obj.elevs[idx] for idx in filt_idxs]
    filt_gr3_obj.elems = [[filt_mapping[node] for node in elem] for elem in filt_gr3_obj.elems]
    filt_gr3_obj.num_elem = len(filt_gr3_obj.elems)
    filt_gr3_obj.num_nodes = len(filt_gr3_obj.lats)
    filt_gr3_obj.elem_coords = [filt_gr3_obj.element_coordinate(i) for i in range(filt_gr3_obj.num_elem)]
    filt_gr3_obj.elem_elevs = [filt_gr3_obj.element_elevations(i) for i in range(filt_gr3_obj.num_elem)]
    node_list_filt = [filt_mapping[node] for node in node_list]
    return filt_gr3_obj, node_list_filt

def parse_gr3_file(gr3_input, filter_open_bnds=False):
    gr3_obj = None
    with open(gr3_input) as f:
        filename = f.readline().rstrip()
        num_elem, num_nodes = f.readline().split()
        gr3_obj = gr3(filename, num_elem, num_nodes)
        prior_node_num = 0
        for i in range(gr3_obj.num_nodes):
            node_num, lon, lat, elev = f.readline().split()
            node_num = int(node_num)
            assert node_num > prior_node_num
            gr3_obj.lons[i] = float(lon)
            gr3_obj.lats[i] = float(lat)
            gr3_obj.elevs[i] = float(elev)
            prior_node_num = node_num
        prior_elem_num = 0
        for i in range(gr3_obj.num_elem):
            elem_num, node_count, *node_lst = f.readline().split()
            elem_num = int(elem_num)
            assert elem_num > prior_elem_num
            gr3_obj.elems[i] = list(map(lambda x: int(x)-1, node_lst))
            prior_elem_num = elem_num
            gr3_obj.elem_coords[i] = gr3_obj.element_coordinate(i)
            gr3_obj.elem_elevs[i] = gr3_obj.element_elevations(i)
        node_list_filt = None
        if filter_open_bnds:
            node_list = []
            num_open_bnds, *extra = f.readline().split()
            num_open_bnds = int(num_open_bnds)
            total_open_bnd_nodes, *extra = f.readline().split()
            total_open_bnd_nodes = int(total_open_bnd_nodes)
            for i in range(num_open_bnds):
                num_nodes, *extra = f.readline().split()
                num_nodes = int(num_nodes)
                for j in range(num_nodes):
                    node_list.append(int(f.readline().rstrip())-1)
            assert len(node_list) == total_open_bnd_nodes
            pre_filt_coords = [(gr3_obj.lons[node],gr3_obj.lats[node], gr3_obj.elevs[node]) for node in node_list]
            gr3_obj, node_list_filt = filter_nodes(gr3_obj, node_list)
            assert len(node_list_filt) == len(node_list)
            assert [(gr3_obj.lons[node],gr3_obj.lats[node], gr3_obj.elevs[node]) for node in node_list_filt] == pre_filt_coords
    return gr3_obj, node_list_filt

def write_nc_mesh(gr3_obj, esmf_output, node_list_filt=None):
    nc = netCDF4.Dataset(esmf_output, "w", format="NETCDF4")
    node_count_dim = nc.createDimension("nodeCount", gr3_obj.num_nodes)
    if node_list_filt:
        node_count_dim = nc.createDimension("openBndNodeCount", len(node_list_filt))
    elem_count_dim = nc.createDimension("elementCount", gr3_obj.num_elem)
    max_elem_len = gr3_obj.max_elem_len()
    max_node_pe_elem_dim = nc.createDimension("maxNodePElement", max_elem_len)
    node_count_dim = nc.createDimension("coordDim", 2)
    node_coords_var = nc.createVariable("nodeCoords","f8",("nodeCount","coordDim"))
    node_coords_var.units = "degrees"
    if node_list_filt:
        bnd_node_var = nc.createVariable("openBndNodes","i","openBndNodeCount")
        bnd_node_var[:] = node_list_filt
    elem_conn_var = nc.createVariable("elementConn","i",("elementCount","maxNodePElement"),fill_value=-1)
    elem_conn_var.long_name = "Node Indices that define the element connectivity"
    num_elem_conn_var = nc.createVariable("numElementConn","b","elementCount")
    num_elem_conn_var.long_name = "Number of nodes per element"
    center_coords_var = nc.createVariable("centerCoords","f8",("elementCount","coordDim"))
    center_coords_var.units = "degrees"
    hgt_node_var = nc.createVariable("Node_Elevations","f8",("nodeCount"))
    hgt_node_var.long_name = "Height above sea level"
    hgt_node_var.units = "meters"
    hgt_node_var[:] = np.where(((np.array(gr3_obj.elevs)-2.75)*-1*0.304) < 0.0,0.0,((np.array(gr3_obj.elevs)-2.75)*-1*0.304))
    hgt_elem_var = nc.createVariable("Element_Elevations","f8",("elementCount"))
    hgt_elem_var.long_name = "Height above sea level"
    hgt_elem_var.units = "meters"
    hgt_elem_var[:] = np.where(((np.array(gr3_obj.elem_elevs)-2.75)*-1*0.304) < 0.0,0.0,((np.array(gr3_obj.elem_elevs)-2.75)*-1*0.304))
    nc.gridType = "unstructured"
    nc.version = "0.9"
    node_coords_var[:, 0] = gr3_obj.lons
    node_coords_var[:, 1] = gr3_obj.lats
    if len(gr3_obj.elems):
        elem_conn_var[:] = gr3_obj.padded_elems()
        num_elem_conn_var[:] = gr3_obj.elem_lens()
        center_coords_var[:] = gr3_obj.elem_coords
    elem_conn_var[:] = elem_conn_var[:] + 1
    nc.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('gr3_input', type=str, help='Input .gr3 file')
    parser.add_argument('esmf_output', type=str, help='Output ESMF .nc file')
    parser.add_argument("--filter_open_bnds", action='store_true', help="Filters only those lines that are open boundaries")
    args = parser.parse_args()
    # Create gr3 class object, which sets up the hgrid.gr3 mesh characteristics and
    # include the offshore water level node boundaries as part of the ESMF mesh file if specified
    gr3_obj, node_list_filt = parse_gr3_file(args.gr3_input, args.filter_open_bnds)
    # Read in hgrid.gr3 object and construct the ESMF mesh netcdf file object
    write_nc_mesh(gr3_obj, args.esmf_output, node_list_filt)


if __name__ == '__main__':
    main()
