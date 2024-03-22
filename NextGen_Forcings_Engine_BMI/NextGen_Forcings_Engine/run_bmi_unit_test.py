"""Run BMI Unit Testing.
Author: jgarrett, modified by jmframe for general/simple nextgen python BMI model
Date: 08/31/2021"""

# TODO: formalize unit tests via typical "unittest" or "pytest" setup

import os
import sys
# import torch
# from torch import nn
from pathlib import Path
import mpi4py
import numpy as np

import bmi_model  # This is the BMI that we will be running

# setup a "success counter" for number of passing and failing bmi functions
# keep track of function def fails (vs function call)
pass_count = 0
fail_count = 0
var_name_counter = 0
fail_list = []

def bmi_except(fstring):
    """Prints message and updates counter and list

    Parameters
    ----------
    fstring : str
        Name of failing BMI function 
    """
    
    global fail_count, fail_list, var_name_counter
    print("**BMI ERROR** in " + fstring)
    if (var_name_counter == 0):
        fail_count += 1
        fail_list.append(fstring)

bmi=bmi_model.NWMv3_Forcing_Engine_BMI_model()

print("\nBEGIN BMI UNIT TEST\n*******************\n");

# Define config path
cfg_file=Path('./config.yml')

if os.path.exists(cfg_file):
    print(" configuration found: " + str(cfg_file))
else:
    print(" no configuration found, exiting...")
    sys.exit()

#-------------------------------------------------------------------
# initialize()
try: 
    bmi.initialize("./config.yml")
    print(" initializing...")
    pass_count += 1
except:
    bmi_except('initialize()')

#-------------------------------------------------------------------
#-------------------------------------------------------------------
# BMI: Model Information Functions
#-------------------------------------------------------------------
#-------------------------------------------------------------------
print("\nMODEL INFORMATION\n*****************")

#-------------------------------------------------------------------
# get_component_name()
try:
    print (" component name: " + bmi.get_component_name())
    pass_count += 1
except:
    bmi_except('get_component_name()')

#-------------------------------------------------------------------
# get_input_item_count()
try:
    print (" input item count: " + str(bmi.get_input_item_count()))
    pass_count += 1
except:
    bmi_except('get_input_item_count()')

#-------------------------------------------------------------------
# get_output_item_count()
try:
    print (" output item count: " + str(bmi.get_output_item_count()))
    pass_count += 1
except:
    bmi_except('get_output_item_count()')

#-------------------------------------------------------------------
# get_input_var_names()
try:    
    # only print statement if names exist
    test_get_input_var_names = bmi.get_input_var_names()
    if len(test_get_input_var_names) > 0:
        print (" input var names: ")
        for var_in in test_get_input_var_names:
            print ("  " + var_in)
    pass_count += 1
except:
    bmi_except('get_input_var_names()')

#-------------------------------------------------------------------
# get_input_var_names()
try:    
    # only print statement if out var list not null
    test_get_output_var_names =  bmi.get_output_var_names()
    if len(test_get_output_var_names) > 0:
        print (" output var names: ")
        for var_out in test_get_output_var_names:
            print ("  " + var_out)
    pass_count += 1
except:
    bmi_except('get_output_item_count()')
    

#-------------------------------------------------------------------
#-------------------------------------------------------------------
# BMI: Variable Information Functions
#-------------------------------------------------------------------
#-------------------------------------------------------------------
print("\nVARIABLE INFORMATION\n********************")

for var_name in (bmi.get_output_var_names() + bmi.get_input_var_names()):  
    print (" " + var_name + ":")

    #-------------------------------------------------------------------
    # get_var_units()
    try: 
        print ("  units: " + bmi.get_var_units(var_name))
        if var_name_counter == 0:
            pass_count += 1
    except:
        bmi_except('get_var_units()')
    
    #-------------------------------------------------------------------
    # get_var_itemsize()
    # JG NOTE: 09.16.2021 AttributeError: 'float' object has no attribute 'dtype'
    try:
        print ("  itemsize: " + str(bmi.get_var_itemsize(var_name)))
        if var_name_counter == 0:
            pass_count += 1
    except:
        bmi_except('get_var_itemsize()')

    #-------------------------------------------------------------------
    # get_var_type()
    # JG NOTE: 09.16.2021 AttributeError: 'float' object has no attribute 'dtype'
    
    # JF NOTE: the print statement needs a string to concatonate
    # JF NOTE: and the type is a native python command.

    try:
        print ("  type: " + str(bmi.get_var_type(var_name)))
        if var_name_counter == 0:
            pass_count += 1
    except:
        bmi_except('get_var_type()')

    #-------------------------------------------------------------------
    # get_var_nbytes()
    # JG NOTE: 09.16.2021 AttributeError: 'float' object has no attribute 'nbytes'
    try:
        print ("  nbytes: " + str(bmi.get_var_nbytes(var_name)))
        if var_name_counter == 0:
            pass_count += 1
    except:
        bmi_except('get_var_nbytes()')

    #-------------------------------------------------------------------
    # get_var_grid
    try:
        print ("  grid id: " + str(bmi.get_var_grid(var_name)))
        if var_name_counter == 0:
            pass_count += 1
    except:
        bmi_except('get_var_grid()')

    #-------------------------------------------------------------------
    # get_var_location
    try:
        print ("  location: " + bmi.get_var_location(var_name))
        if var_name_counter == 0:
            pass_count += 1
    except:
        bmi_except('get_var_location()')

    var_name_counter += 1

# reset back to zero
var_name_counter = 0

#-------------------------------------------------------------------
#-------------------------------------------------------------------
# BMI: Time Functions
#-------------------------------------------------------------------
#-------------------------------------------------------------------
print("\nTIME INFORMATION\n****************")

#-------------------------------------------------------------------
# get_start_time()
try:
    print (" start time: " + str(bmi.get_start_time()))
    pass_count += 1
except:
    bmi_except('get_start_time()')

#-------------------------------------------------------------------
# get_end_time()
try:
    print (" end time: " + str(bmi.get_end_time()))
    pass_count += 1
except:
    bmi_except('get_end_time()')

#-------------------------------------------------------------------
# get_current_time()
try:
    print (" current time: " + str(bmi.get_current_time()))
    pass_count += 1
except:
    bmi_except('get_current_time()')

#-------------------------------------------------------------------
# get_time_step()
try:
    print (" time step: " + str(bmi.get_time_step()))
    pass_count += 1
except:
    bmi_except('get_time_step()')

#-------------------------------------------------------------------
# get_time_units()
try:
    print (" time units: " + bmi.get_time_units())
    pass_count += 1
except:
    bmi_except('get_time_units()')


#-------------------------------------------------------------------
#-------------------------------------------------------------------
# BMI: Model Grid Functions
#-------------------------------------------------------------------
#-------------------------------------------------------------------
print("\nGRID INFORMATION\n****************")

for grid in bmi._grids:

    grid_id = grid._id

    print (" grid id: " + str(grid_id))

    #-------------------------------------------------------------------
    # get_grid_rank()
    try:
        print ("  rank: " + str(bmi.get_grid_rank(grid_id)))
        pass_count += 1
    except:
        bmi_except('get_grid_rank()')

    #-------------------------------------------------------------------
    # get_grid_size()
    try:    
        print ("  size: " + str(bmi.get_grid_size(grid_id)))
        pass_count += 1
    except:
        bmi_except('get_grid_size()')

    #-------------------------------------------------------------------
    # get_grid_type()    
    try:
        print ("  type: " + bmi.get_grid_type(grid_id))
        pass_count += 1
    except:
        bmi_except('get_grid_type()')    

    #-------------------------------------------------------------------
    # get_grid_x()
    try:
        print ("  grid x: " + str(bmi.get_grid_x(grid_id)))
        pass_count += 1
    except:
        bmi_except('get_grid_x()')

    #-------------------------------------------------------------------
    # get_grid_y()
    try:
        print ("  grid y: " + str(bmi.get_grid_y(grid_id)))
        pass_count += 1
    except:
        bmi_except('get_grid_y()')

    if(bmi._grid_type == "gridded"):
        #-------------------------------------------------------------------
        # get_grid_shape()
        try:
            print ("  grid shape: " + str(bmi.get_grid_shape(grid_id)))
            pass_count += 1
        except:
            bmi_except('get_grid_shape()')

        #-------------------------------------------------------------------
        # get_grid_spacing()
        try:
            print ("  grid spacing: " + str(bmi.get_grid_spacing(grid_id)))
            pass_count += 1
        except:
            bmi_except('get_grid_spacing()')

        #-------------------------------------------------------------------
        # get_grid_origin()
        try:
            print ("  grid origin: " + str(bmi.get_grid_origin(grid_id)))
            pass_count += 1
        except:
            bmi_except('get_grid_origin()')

    if(bmi._grid_type != "gridded"):
        #-------------------------------------------------------------------
        # get_grid_edge_count()
        try:
            print ("  grid edge count: " + str(bmi.get_grid_edge_count(grid_id)))
            pass_count += 1
        except:
            bmi_except('get_grid_edge_count()')

        #-------------------------------------------------------------------
        # get_grid_node_count()
        try:
            print ("  grid node count: " + str(bmi.get_grid_node_count(grid_id)))
            pass_count += 1
        except:
            bmi_except('get_grid_node_count()')

        #-------------------------------------------------------------------
        # get_grid_face_count()
        try:
            print ("  grid face count: " + str(bmi.get_grid_face_count(grid_id)))
            pass_count += 1
        except:
            bmi_except('get_grid_face_count()')

        #-------------------------------------------------------------------
        # get_grid_edge_nodes()
        try:
            print ("  grid edge nodes: " + str(bmi.get_grid_edge_nodes(grid_id)))
            pass_count += 1
        except:
            bmi_except('get_grid_edge_nodes()')

        #-------------------------------------------------------------------
        # get_grid_face_edges()
        try:
            print ("  grid face edges: " + str(bmi.get_grid_face_edges(grid_id)))
            pass_count += 1
        except:
            bmi_except('get_grid_face_edges()')

        #-------------------------------------------------------------------
        # get_grid_face_nodes()
        try:
            print ("  grid face nodes: " + str(bmi.get_grid_face_nodes(grid_id)))
            pass_count += 1
        except:
            bmi_except('get_grid_face_nodes()')

        #-------------------------------------------------------------------
        # get_grid_nodes_per_face()
        try:
            print ("  grid nodes per face: " + str(bmi.get_grid_nodes_per_face(grid_id)))
            pass_count += 1
        except:
            bmi_except('get_grid_nodes_per_face()')


#-------------------------------------------------------------------
#-------------------------------------------------------------------
# BMI: Update in time once so we can test ensure the GET Values functions
# can extract raw regridded data properly
# update()
try:
    bmi.update()
    # go ahead and print time to show iteration
    # TODO: this will fail if get_current_time() does
    print (" updating...        time " + str(bmi.get_current_time()))
    pass_count += 1
except:
    bmi_except('update()')

#-------------------------------------------------------------------
#-------------------------------------------------------------------
# BMI: Variable Getter and Setter Functions
#-------------------------------------------------------------------
#-------------------------------------------------------------------    
print ("\nGET VALUES\n******************")

# TODO: 09.16.2021 this is a band-aid fix for how lstm handles input vars rn
for var_name in (bmi.get_output_var_names()):     
    print (" " + var_name + ":" )
    #-------------------------------------------------------------------
    # get_value_ptr()
    try:
        #print ("  get value ptr: {:.2f}".format(bmi.get_value_ptr(var_name)))
        print ("  get value ptr: " + str(bmi.get_value_ptr(var_name)))
        if var_name_counter == 0: 
            pass_count += 1
    except:
        bmi_except('get_value_ptr()')

    #-------------------------------------------------------------------
    # get_value()
    try:
        dest0 = np.empty(bmi.get_grid_size(bmi.get_var_grid(var_name)), dtype=float)
        #print ("  get value: {:.2f}".format(bmi.get_value(var_name)))
        print ("  get value: " + str(bmi.get_value(var_name,dest0)))
        if var_name_counter == 0: 
            pass_count += 1
    except:
        bmi_except('get_value()')

    #-------------------------------------------------------------------
    # get_value_at_indices()    
    try: 
        dest0 = np.empty(bmi.get_grid_size(bmi.get_var_grid(var_name)), dtype=float)
        # JMFrame NOTE: converting a list/array to a string probably won't work
        #print ("  get value at indices: " + str(bmi.get_value_at_indices(var_name, dest0, [0])))
        
        print ("  get value at indices: ", bmi.get_value_at_indices(var_name, dest0, np.array([0])))
        
        if var_name_counter == 0: 
            pass_count += 1
    except: 
        bmi_except('get_value_at_indices()')

    var_name_counter += 1

# set back to zero
var_name_counter = 0

#-------------------------------------------------------------------
#-------------------------------------------------------------------
# BMI: Control Functions
#-------------------------------------------------------------------
#-------------------------------------------------------------------   
print ("\nCONTROL FUNCTIONS\n*****************")    
    
#-------------------------------------------------------------------
# update()
try:
    bmi.update()
    # go ahead and print time to show iteration
    # TODO: this will fail if get_current_time() does
    print (" updating...        time " + str(bmi.get_current_time()))
    pass_count += 1
except:
    bmi_except('update()')

#-------------------------------------------------------------------
# update_until()
try:
    bmi.update_until(36000)
    # go ahead and print time to show iteration
    # TODO: this will fail if get_current_time() does
    print (" updating until...  time ", bmi.get_current_time())
    pass_count += 1
except:
    bmi_except('update_until()')          

#-------------------------------------------------------------------
# finalize()
try:
    bmi.finalize()
    print (" finalizing...")
    pass_count += 1
except:
    bmi_except('finalize()')

# lastly - print test summary
print ("\n Total BMI function PASS: " + str(pass_count))
print (" Total BMI function FAIL: " + str(fail_count))
for ff in fail_list:
    print ("  " + ff)   
