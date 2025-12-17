import numpy as np
import netCDF4 as nc

def binary_isin(elements, test_elements, assume_sorted=False, return_indices=False):
    """Test if values of elements are present in test_elements.
    Returns a boolean array: True if element is present in test_elements, False otherwise.

    If return_indices=True, return an array of indexes that transforms the elements of
    test_elements to same order of elements (for the elements that are present in test_elements).
    ie, for every True in the returned mask, also return the index such that test_elements[indices] == elements[mask]

    The method is usually slower than using np.isin for unsorted test_elements.
    However, the returns of this method can still be useful.
    """
    elements = np.asarray(elements,dtype=int)
    test_elements = np.asarray(test_elements,dtype=int)

    if assume_sorted:
        idxs = np.searchsorted(test_elements, elements)
    else:
        asorted = test_elements.argsort()
        idxs = np.searchsorted(test_elements, elements, sorter=asorted)

    valid = idxs != len(test_elements)
    test_selector = idxs[valid]
    if not assume_sorted:
        test_selector = asorted[test_selector]

    mask = np.zeros(elements.shape, dtype=bool)
    mask[valid] = test_elements[test_selector] == elements[valid]

    #indices are the array of indexes that transform the elements in
    # test_elements to match the order of elements
    if return_indices:
        return mask, test_selector[mask[valid]]
    else:
        return mask

def extract_lat_lon(current_netcdf_filename, idxs):
    """Extract latitude and longitude from a netcdf file for a set of stations.
    
    Args:
        current_netcdf_filename: path to netcdf
        idxs: indexes of stations to read
    
    Returns:
        (tuple): ndarrays for latitude and longitude respectively.
    """

    with nc.Dataset(current_netcdf_filename) as ncdata:
        # extract the streamflow
        lat_vals = ncdata['latitude'][idxs]
        lon_vals = ncdata['longitude'][idxs]
        return (lat_vals, lon_vals)

def extract_offsets(current_netcdf_filename, selected_ids):
    """Return a mask on feature_id for the selected_ids

    Args:
        current_netcdf_filename (str): NetCDF file to read
        selected_ids (list_like): ids to select.

    Returns:
        (ndarray): Indexes *in order* of selected comm_ids.
    """
    with nc.Dataset(current_netcdf_filename) as ncdata:
        feature_id_index = ncdata['feature_id'][:]
        print("Size of feature id data is ", len(feature_id_index))
        fmask, fidx = binary_isin(feature_id_index, selected_ids, return_indices=True)
        fvals = feature_id_index[fmask]
        return fmask, fvals, fidx
