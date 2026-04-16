import cv2
import numpy as np
from geopy.distance import geodesic
from pyproj import Transformer
import math

def transform_image_to_utm(correspondences, query_coords):
    """
    Relates image pixels to UTM coordinates using a normalized homography.
    
    Args:
        correspondences: List of tuples [((u, v), (east, north)), ...] 
                         Requires at least 4 pairs.
        query_coords: np.array of shape (N, 2) containing pixels to transform.
        
    Returns:
        np.array: Transformed UTM coordinates of shape (N, 2).
    """
    assert len(correspondences) >= 4, f"Computing homography requires at least 4 pairs of correspondences, get {len(correspondences)} instead"

    src_pts = np.array([c[0] for c in correspondences], dtype=np.float32)
    dst_pts_raw = np.array([c[1] for c in correspondences], dtype=np.float64)

    local_origin = np.mean(dst_pts_raw, axis=0)
    dst_pts_norm = (dst_pts_raw - local_origin).astype(np.float32)

    H, _ = cv2.findHomography(src_pts, dst_pts_norm)
    query_pts = np.array(query_coords, dtype=np.float32).reshape(-1, 1, 2)
    
    transformed_norm = cv2.perspectiveTransform(query_pts, H)
    
    utm_results = transformed_norm.reshape(-1, 2) + local_origin

    return utm_results

def transform_wgs84_to_utm(coords):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32648", always_xy=True)
    utm_coord = []
    for coord in coords:
        utm_coord.append(transformer.transform(coord[0], coord[1]))
    
    return np.array(utm_coord)

def distance_from_coordinates(
    coord1, coord2, crs:str="UTM"
):
    """
    Calculate distance (in meter) between 2 locations
    """
    assert crs.upper() in {"WGS84", "UTM"}, f"Does not support distance computation with {crs} coordinate"
    
    if crs.upper() == "WGS84":
        distance = geodesic(coord1, coord2).meters

    elif crs.upper() == "UTM":
        distance = math.dist(coord1, coord2)

    return distance