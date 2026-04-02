from geopy.distance import geodesic

def distance_from_coordinates(
        lon_x: float,
        lat_x: float,
        lon_y: float,
        lat_y: float
):
    """
    Calculate distance (in meter) between 2 locations
    """
    coord1 = (lon_x, lat_x)
    coord2 = (lon_y, lat_y)

    distance = geodesic(coord1, coord2).meters

    return distance

