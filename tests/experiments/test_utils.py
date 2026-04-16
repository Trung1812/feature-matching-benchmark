from feature_matching.experiments.utils import *
import pytest

landmark_dir_test = Path("/home/c4i/workspace/trungpq16/uav220/feature_matching_experiments/tests/data/landmark/test")

@pytest.mark.parametrize("x,y,landmark_dir,output",
                     [
                        (550892.4628019021,2326025.885153106,landmark_dir_test, True),
                        (568946.0, 2425436.1235, landmark_dir_test, False)
                     ])
def test_is_good_query_true(x, y, landmark_dir, output):
    cover = get_cover(landmark_dir)
    assert is_good_query(x, y, cover) == output
    
uav_path = landmark_dir_test / "00_15_44.jpg"
@pytest.mark.parametrize("uav_path, expected", [
    (uav_path, (550892.4628019021, 2326025.885153106))
])
def test_get_uav_center(uav_path, expected):
    assert get_uav_center(uav_path) == expected