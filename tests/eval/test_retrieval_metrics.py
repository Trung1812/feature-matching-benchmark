import pytest
from feature_matching.eval.retrieval_metrics import distance_from_coordinates

#TODO: Populate the test
@pytest.mark.parametrize(
    "lon_x, lat_x, lon_y, lat_y, distance", 
    [(52.2296756, 21.0122287, 52.406374, 16.9251681, 279352.901604)
     ]
)

def test_distance_from_coordinates(
    lon_x, lat_x, lon_y, lat_y, distance
):
    epsilon = 0.1
    assert abs(distance_from_coordinates(lon_x, lat_x, lon_y, lat_y) - distance) < epsilon
