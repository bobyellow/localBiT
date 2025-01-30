import numpy as np
from collections import defaultdict
from scipy.spatial import KDTree
import shapefile

# Define function for Queen's and Rook's contiguity neighbors
def getNeighborsAreaContiguity(AREAS):
    """
    Generates contiguity-based neighbor lists (Queen and Rook).

    This function returns the Wrook and Wqueen dictionaries from a set of areas.

    :param AREAS: Set of polygons to calculate the neighbors.
    :type AREAS: list

    :return: Tuple of dictionaries (Wqueen, Wrook) containing neighbor relationships.
    :rtype: (dict, dict)
    """
    segment2areas = {}
    point2areas = {}
    Wqueen = {}
    Wrook = {}

    # Initialize neighbor dictionaries
    for idx in range(len(AREAS)):
        Wqueen[idx] = []
        Wrook[idx] = []

    for a, area in enumerate(AREAS):
        for ring in area:
            for p, point in enumerate(ring[:-1]):  # Exclude the last point (same as the first)
                # Define points and segments
                p1 = tuple(round(coord, 3) for coord in point)
                p2 = tuple(round(coord, 3) for coord in ring[p + 1])
                segment = sorted([p1, p2], key=lambda x: x[0]**2 + x[1]**2)
                sortSegment = tuple(segment)

                # Update Rook neighbors (shared edges)
                if sortSegment in segment2areas:
                    segment2areas[sortSegment].append(a)
                    areasRook = segment2areas[sortSegment]
                    for area1 in areasRook:
                        for area2 in areasRook:
                            if area2 not in Wrook[area1] and area2 != area1:
                                Wrook[area1].append(area2)
                                Wrook[area2].append(area1)
                else:
                    segment2areas[sortSegment] = [a]

                # Update Queen neighbors (shared points)
                if p1 in point2areas:
                    point2areas[p1].append(a)
                    areasQueen = point2areas[p1]
                    for area1 in areasQueen:
                        for area2 in areasQueen:
                            if area2 not in Wqueen[area1] and area2 != area1:
                                Wqueen[area1].append(area2)
                                Wqueen[area2].append(area1)
                else:
                    point2areas[p1] = [a]

    return Wqueen, Wrook

# Define function for k-nearest neighbors
def kNearestNeighbors(centroids, k):
    """
    Calculate k-nearest neighbors based on centroids.

    :param centroids: List of (x, y) coordinates representing spatial unit centroids.
    :type centroids: list of tuples

    :param k: Number of nearest neighbors to calculate.
    :type k: int

    :return: Dictionary where keys are unit indices and values are lists of neighbor indices.
    :rtype: dict
    """
    tree = KDTree(centroids)
    neighbors = {}

    for i, centroid in enumerate(centroids):
        # Query the k+1 nearest neighbors (including itself)
        distances, indices = tree.query(centroid, k=k+1)
        neighbors[i] = indices[1:].tolist()  # Exclude itself (first result)

    return neighbors

# Extract centroids from a polygon shapefile
def extractCentroidsFromShapefile(shapefile_path):
    """
    Extract centroids from a polygon shapefile.

    :param shapefile_path: Path to the input shapefile.
    :type shapefile_path: str

    :return: List of centroids as (x, y) coordinates.
    :rtype: list of tuples
    """
    sf = shapefile.Reader(shapefile_path)
    centroids = []

    for shape in sf.shapes():
        # Calculate the centroid as the mean of all points in the polygon
        points = shape.points
        centroid = tuple(np.mean(points, axis=0))
        centroids.append(centroid)

    return centroids
