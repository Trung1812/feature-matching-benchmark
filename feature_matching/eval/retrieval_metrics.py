from geopy.distance import geodesic
from pathlib import Path
import pandas as pd
import json

def distance_from_coordinates(
    lon_x: float,
    lat_x: float,
    lon_y: float,
    lat_y: float
):
    """
    Calculate distance (in meter) between 2 locations
    """
    coord1 = (lat_x, lon_x)
    coord2 = (lat_y, lon_y)

    distance = geodesic(coord1, coord2).meters

    return distance

def calculate_center_coord(
    frame_corner_coordinates: tuple[float]
):
    """
    Calculate center pixel's coordinate in WGS84

    Params
    ------
    frame_corner_coordinates: coordinates (lon, lat) of the frame's corners (upper left, upper right, lower right, lower left)

    Return
    ------
    tuple[float]
    """
    if len(frame_corner_coordinates) != 4:
        raise ValueError("frame_corner_coordinate must contain exactly 4 corners")

    lon_sum = 0.0
    lat_sum = 0.0

    for corner in frame_corner_coordinates:
        if not isinstance(corner, (tuple, list)) or len(corner) != 2:
            raise ValueError("Each corner must be a (lon, lat) pair")
        lon_sum += float(corner[0])
        lat_sum += float(corner[1])

    return (lon_sum / 4.0, lat_sum / 4.0)

def classify_result(   
    lon_pred: float,
    lat_pred: float,
    lon_gt: float,
    lat_gt: float,
    tolerance: float = 5,
    is_good_query: bool = False
    ) -> str:
    """
    Classify result into TP, FP1, FP2, TN, FN (read docs for meaning).
    Params
    ------ 
    """
    is_valid_prediction = bool(lon_pred) and bool(lat_pred)

    if not is_good_query:
        
        if is_valid_prediction:
            return "FP2"
        else:
            return "TN"
    else:
        if not is_valid_prediction:
            return "FN"
        else:
            error = distance_from_coordinates(lon_pred, lat_pred, lon_gt, lat_gt)
            if error <= tolerance:
                return "TP"
            else:
                return "FP1"


def load_results(results_dir: Path) -> pd.DataFrame:   
    """Load multiple json files with same structure to a DataFrame for analysis"""
    result_json_paths = sorted(results_dir.glob("*.json"))

    rows: list[dict] = []
    for json_path in result_json_paths:
        with json_path.open("r", encoding="utf-8") as f:
            result = json.load(f)

        if isinstance(result, list):
            rows.extend(item for item in result if isinstance(item, dict))
        elif isinstance(result, dict):
            rows.append(result)

    df = pd.DataFrame(rows)
    
    return df

def classify_result_on_df(df: pd.DataFrame, tolerance: float):
    col_name = f"retrieval_result@{tolerance:g}m"
    df[col_name] = df.apply(
        lambda row: classify_result(
            lon_pred=row["lon_pred"],
            lat_pred=row["lat_pred"],
            lon_gt=row["lon_gt"],
            lat_gt=row["lat_gt"],
            tolerance=tolerance,
            is_good_query=row["is_good_query"],
        ),
        axis=1,
    )

    return df
    
def calculate_precision(df: pd.DataFrame, tolerance: float) -> float:
    res_col_name = f'retrieval_result@{tolerance:g}m' 
    if res_col_name not in df.columns:
        classify_result_on_df(df, tolerance)
    
    true_positive = df[res_col_name].str.count("TP").sum()
    false_positive = df[res_col_name].str.count("FP").sum()

    return true_positive / (true_positive + false_positive)

def calculate_recall(df: pd.DataFrame, tolerance: float) -> float:
    res_col_name = f'retrieval_result@{tolerance:g}m'
    if res_col_name not in df.columns:
        classify_result_on_df(df, tolerance)

    true_positive = df[res_col_name].str.count("TP").sum()
    false_negative = df[res_col_name].str.count("FN").sum()

    return true_positive / (true_positive + false_negative)
    


    
    
    
    

