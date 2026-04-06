import pytest
import json

from feature_matching.eval.retrieval_metrics import (
    calculate_center_coord,
    distance_from_coordinates,
    load_results,
)

#TODO: Populate the test
@pytest.mark.parametrize(
    "lat_x, lon_x, lat_y, lon_y, distance", 
    [(52.2296756, 21.0122287, 52.406374, 16.9251681, 279352.901604)
     ]
)

def test_distance_from_coordinates(
    lon_x, lat_x, lon_y, lat_y, distance
):
    epsilon = 0.1
    assert abs(distance_from_coordinates(lon_x, lat_x, lon_y, lat_y) - distance) < epsilon


def test_calculate_center_coord():
    corners = (
        (105.0, 21.0),
        (107.0, 21.0),
        (107.0, 23.0),
        (105.0, 23.0),
    )

    center = calculate_center_coord(corners)

    assert center == (106.0, 22.0)


def test_load_results(tmp_path):
    sample_1 = {
        "lon_pred": 105.0,
        "lat_pred": 21.0,
        "lon_gt": 105.0,
        "lat_gt": 21.0,
        "is_good_query": True,
    }
    sample_2 = {
        "lon_pred": 0,
        "lat_pred": 0,
        "lon_gt": 105.5,
        "lat_gt": 21.5,
        "is_good_query": False,
    }

    (tmp_path / "a.json").write_text(json.dumps(sample_1), encoding="utf-8")
    (tmp_path / "b.json").write_text(json.dumps(sample_2), encoding="utf-8")

    df = load_results(tmp_path, tolerance=5.0)

    assert len(df) == 2
    assert "retrieval_result@5m" in df.columns
    assert list(df["retrieval_result@5m"]) == ["TP", "TN"]
