from shapely.geometry import MultiPoint, Polygon, Point
from scipy.spatial import cKDTree


def clip_point_to_roi(vertices, points):
    """Clip points to a polygon region of interest

    Arguments:
      vertices: Ordered array of vertices that describe polygon
      points: an Nx2 array of points

    Returns:
      (list): Boolean mask of size N. True if point lies inside polygon
        False otherwise.
    """
    rv = []
    polygon = Polygon(vertices)
    multipoly = MultiPoint(points)
    for point in range(len(multipoly.geoms)):
        pt = (multipoly.geoms[point].x,multipoly.geoms[point].y)
        rv.append(polygon.contains(Point(pt)))
    return rv
    #for pt in MultiPoint(points):
    #    rv.append(polygon.contains(pt))
    #return rv

def kd_nearest_neighbor(vertices, points):
    """Find nearest neighbors using KDtree

    Args:
        vertices (ndarray): A Nx2 array of vertices
        points (ndarray): points to use to find nearest neighbors

    Returns:
        [type]: Nearest neighbors
    """
    tree = cKDTree(vertices)
    return tree.query(points)
