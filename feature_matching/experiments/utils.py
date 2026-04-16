import json
from pathlib import Path
from rtree import index
import numpy as np

def load_metadata(meta_path: Path) -> dict:
    if not meta_path.exists():
        return {}
    with open(meta_path, "r", encoding="utf8") as f:
        return json.load(f)

def get_uav_center(uav_path: Path):
    metadata = load_metadata(uav_path.with_suffix(".json"))
    loc = metadata.get("epsg_3805", None)
    if loc is None:
        return None, None
    else:
        x = float(loc.get("center_x"))
        y = float(loc.get("center_y"))
        return (x, y)

def get_cover(landmark_dir: Path):
    """
    Load metadata and build an R-tree index for high-efficiency spatial lookups.
    """
    def parse(metadata):
        loc = metadata.get("epsg_3805", None)
        if loc is None:
            return None
        else:
            min_x = loc.get("upper_left_x")
            max_x = loc.get("upper_right_x")

            min_y = loc.get("down_left_y")
            max_y = loc.get("upper_left_y")
        
        return tuple(map(float,(min_x, min_y, max_x, max_y)))

    cover = index.Index()
    meta_paths = landmark_dir.glob("*.json")

    for idx, meta_path in enumerate(meta_paths):
        metadata = load_metadata(meta_path)
        bbox = parse(metadata)
        if bbox is not None:
            cover.insert(idx, bbox)
       
    return cover

def is_good_query(x, y, cover):
    """
    Checks if a point is contained within any area in the R-tree cover.
    center_loc: (x, y)
    cover: An rtree.index.Index object
    """
    if x is None or y is None:
        return False
    return cover.count((x, y, x, y)) > 0

def save_result(file_path: Path, uav_path: Path, result: dict):
    # Check if file exists and is not empty
    if file_path.exists() and file_path.stat().st_size > 0:
        with open(file_path, 'r') as f:
            try:
                data_list = json.load(f)
            except json.JSONDecodeError:
                data_list = []
    else:
        data_list = []

    if not isinstance(data_list, list):
        data_list = [data_list]

    data_list.append(result)
    with open(file_path, 'w') as f:
        json.dump(data_list, f, indent=4)
    
def get_landmark_translation_and_homography(lm_path: Path):
    metadata = load_metadata(lm_path.with_suffix(".json"))

    homography = np.array(metadata.get("H_norm"))
    translation = np.array(metadata.get("T"))
    return homography, translation