# Overview
This subdirectory with the NextGen Forcings BMI Github repository contains Python scripts that are only focused on converting model domain file formats into a ESMF mesh compliant netcdf file that can be directly utilized by the NextGen Forcings Engine BMI. So far, this repository contains scripts to convert a NextGen hydrofabric geopackage or coastal model mesh file inputs (D-FlowFM, SCHISM) into ESMF mesh compliant netcdf files. Future updates to this repository will reflect more NextGen model formulations as they become available. This sub-directory is now directly used as a Python module to streamline pre-processing steps of the NextGen Forcings Engine BMI, or within it's direct execution. 

# Setting Up Required Python Environment to Execute Forcing Extraction Scripts Using Anaconda
`conda env create --name ngen_esmf_mesh_prod --file=environment.yml`

# Ongoing Evaluations
We've recently resolved a slew of issues to ensure that this ESMF Mesh production script works across all versions of the NextGen hydrofabric. We have been able to resolve the following issues that we have found below:

Root Causes:
1. MultiPolygon/GeometryCollection Incompatibilities: ESMF elements must be
   single contiguous polygons. Passing MultiPolygons caused missing `.exterior` 
   AttributeErrors in Python or degenerate triangulation in ESMF.
2. Micro-Spikes & Hairpin Vertices: GIS vector operations generated 180-degree
   hairpin U-turns (vertices doubling back on themselves) and near-duplicate
   points. While valid under 2D planar Shapely, these produce 0-area vectors and
   cross-product cancellations during 3D spherical Delaunay triangulation.
3. Self-Intersection Artifacts: Invalid 2D topologies (e.g., bowties) and 
   clockwise winding orders led to negative spherical area calculations.

Resolution:
- Implemented `extract_largest_polygon()` helper to guarantee all Shapely 
  outputs (`make_valid`, `set_precision`, `simplify`) strictly reduce back to
  a single `Polygon` instance.
- Added `clean_geometry_for_esmf()` pipeline to sanitize hydrofabric features:
  * Snaps vertices to a ~10cm grid (`grid_size=1e-6`) to merge micro-duplicates.
  * Simplifies topology to collapse hairpin U-turns and collinear points.
  * Applies `shapely.make_valid()` for self-intersecting bowties.
  * Enforces counter-clockwise (CCW) winding order for positive ESMF surface normals.
- Refactored geometry processing loops to ensure safe node extraction and 
  centroid allocation.
