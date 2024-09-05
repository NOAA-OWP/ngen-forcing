import argparse
import netCDF4
import numpy as np
import shapely
from pyproj import Geod

"""
Script to perform a conversion between an D-Flow FM mesh file and ESMF Unstructured Grid Format

Example Usage:  ./dflowfm_2_esmf.py DFlowFM_Mesh.nc DFlowFM_ESMF_Mesh.nc
"""

def write_nc_mesh_config2(dflowfm_mesh_file, esmf_output):

    # Read in mesh characteristics and set up ESMF netcdf file formatting
    with netCDF4.Dataset(dflowfm_mesh_file) as f:

        node_lat = f.variables['NetNode_y'][:].data
        node_lon = f.variables['NetNode_x'][:].data
        node_hgt = f.variables['NetNode_z'][:].data
        elem_conn = f.variables['NetElemNode'][:].data
        node_conn = f.variables['NetElemNode'][:]

        nc = netCDF4.Dataset(esmf_output, "w", format="NETCDF4")
        node_count_dim = nc.createDimension("nodeCount", f.dimensions['nNetNode'].size)
        elem_count_dim = nc.createDimension("elementCount", f.dimensions['nNetElem'].size)
        max_node_pe_elem_dim = nc.createDimension("maxNodePElement", f.dimensions['nNetElemMaxNode'].size)
        node_count_dim = nc.createDimension("coordDim", 2)
        node_coords_var = nc.createVariable("nodeCoords","f8",("nodeCount","coordDim"))
        node_coords_var.units = "degrees"
        elem_conn_var = nc.createVariable("elementConn","i",("elementCount","maxNodePElement"),fill_value=-1)
        elem_conn_var.long_name = "Node Indices that define the element connectivity"
        num_elem_conn_var = nc.createVariable("numElementConn","i","elementCount")
        num_elem_conn_var.long_name = "Number of nodes per element"
        center_coords_var = nc.createVariable("centerCoords","f8",("elementCount","coordDim"))
        center_coords_var.units = "degrees"
        hgt_node_var = nc.createVariable("Node_Elevations","f8",("nodeCount"))
        hgt_node_var.long_name = "Height above sea level"
        hgt_node_var.units = "meters"
        hgt_elem_var = nc.createVariable("Element_Elevations","f8",("elementCount"))
        hgt_elem_var.long_name = "Height above sea level"
        hgt_elem_var.units = "meters"
        nc.gridType = "unstructured"
        nc.version = "0.9"
        node_coords_var[:, 0] = node_lon
        node_coords_var[:, 1] = node_lat
        hgt_node_var[:] = node_hgt
        geod = Geod(ellps="WGS84")

        # Loop to rotate the nodes of an element if they're counterclockwise
        for i in range(f.dimensions['nNetElem'].size):
            num_nodes = np.ma.count(node_conn[i,:])
            num_nodes_indices = node_conn[i,0:num_nodes]-1
            hgt_elem_var[i] = np.sum(node_hgt[num_nodes_indices])/num_nodes
            center_coords_var[i,0] = np.sum(node_lon[num_nodes_indices])/num_nodes
            center_coords_var[i,1] = np.sum(node_lat[num_nodes_indices])/num_nodes
            num_elem_conn_var[i] = num_nodes
            node_x = node_coords_var[elem_conn[i,0:num_nodes]-1,0]
            node_y = node_coords_var[elem_conn[i,0:num_nodes]-1,1]
            node_x = np.append(node_x,node_x[0])
            node_y = np.append(node_y,node_y[0])
            polygon = shapely.Polygon(np.column_stack([node_x,node_y]))
            poly_area, poly_perimeter = geod.geometry_area_perimeter(polygon)
            if(poly_area < 0.0):
                elem_conn[i,0:num_nodes] = np.flip(elem_conn[i,0:num_nodes])
                print('element id that was cw is  ', str(i+1))
                print(poly_area)
        elem_conn = np.where(elem_conn<0,-1,elem_conn)
        elem_conn_var[:] = elem_conn

    nc.close()

def write_nc_mesh_config1(dflowfm_mesh_file, esmf_output):

    # Read in mesh characteristics and set up ESMF netcdf file formatting
    with netCDF4.Dataset(dflowfm_mesh_file) as f:
        node_lat = f.variables['mesh2d_node_y'][:].data
        node_lon = f.variables['mesh2d_node_x'][:].data
        node_hgt = f.variables['mesh2d_node_z'][:].data
        elem_conn = f.variables['mesh2d_face_nodes'][:].data
        elem_x = f.variables['mesh2d_face_x'][:].data
        elem_y = f.variables['mesh2d_face_y'][:].data
        node_conn = f.variables['mesh2d_face_nodes'][:]

        nc = netCDF4.Dataset(esmf_output, "w", format="NETCDF4")
        node_count_dim = nc.createDimension("nodeCount", len(node_lat))
        elem_count_dim = nc.createDimension("elementCount", len(elem_x))
        max_node_pe_elem_dim = nc.createDimension("maxNodePElement", node_conn.shape[-1])
        node_count_dim = nc.createDimension("coordDim", 2)
        node_coords_var = nc.createVariable("nodeCoords","f8",("nodeCount","coordDim"))
        node_coords_var.units = "degrees"
        elem_conn_var = nc.createVariable("elementConn","i",("elementCount","maxNodePElement"),fill_value=-1)
        elem_conn_var.long_name = "Node Indices that define the element connectivity"
        num_elem_conn_var = nc.createVariable("numElementConn","i","elementCount")
        num_elem_conn_var.long_name = "Number of nodes per element"
        center_coords_var = nc.createVariable("centerCoords","f8",("elementCount","coordDim"))
        center_coords_var.units = "degrees"
        hgt_node_var = nc.createVariable("Node_Elevations","f8",("nodeCount"))
        hgt_node_var.long_name = "Height above sea level"
        hgt_node_var.units = "meters"
        hgt_elem_var = nc.createVariable("Element_Elevations","f8",("elementCount"))
        hgt_elem_var.long_name = "Height above sea level"
        hgt_elem_var.units = "meters"
        nc.gridType = "unstructured"
        nc.version = "0.9"
        node_coords_var[:, 0] = node_lon
        node_coords_var[:, 1] = node_lat
        hgt_node_var[:] = node_hgt
        center_coords_var[:,0] = elem_x
        center_coords_var[:,1] = elem_y
        geod = Geod(ellps="WGS84")

        # Loop to rotate the nodes of an element if they're counterclockwise
        for i in range(len(elem_x)):
            num_nodes = np.ma.count(node_conn[i,:])
            num_nodes_indices = node_conn[i,0:num_nodes]-1
            hgt_elem_var[i] = np.sum(node_hgt[num_nodes_indices])/num_nodes
            num_elem_conn_var[i] = num_nodes
            node_x = node_coords_var[elem_conn[i,0:num_nodes]-1,1]
            node_y = node_coords_var[elem_conn[i,0:num_nodes]-1,0]
            node_x = np.append(node_x,node_x[0])
            node_y = np.append(node_y,node_y[0])
            polygon = shapely.Polygon(np.column_stack([node_x,node_y]))
            poly_area, poly_perimeter = geod.geometry_area_perimeter(polygon)
            if(poly_area < 0.0):
                elem_conn[i,0:num_nodes] = np.flip(elem_conn[i,0:num_nodes])
                print('element id that was cw is ', str(i+1))
                print(poly_area)
        elem_conn = np.where(elem_conn<0,-1,elem_conn)
        elem_conn_var[:] = elem_conn

    nc.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dflowfm_netcdf_mesh_input', type=str, help='Input dflowfm mesh netcdf file')
    parser.add_argument('esmf_output', type=str, help='Output ESMF compliant netcdf file')
    args = parser.parse_args()
    # Attempt both DFlowFM ESMF mesh netcdf file construction
    # methods, which differ by the naming convention of the
    # DFlowFM mesh characteristics
    try:
        dflowfm_obj = write_nc_mesh_config1(args.dflowfm_netcdf_mesh_input,args.esmf_output)
    except:
        dflowfm_obj = write_nc_mesh_config2(args.dflowfm_netcdf_mesh_input,args.esmf_output)



if __name__ == '__main__':
    main()
