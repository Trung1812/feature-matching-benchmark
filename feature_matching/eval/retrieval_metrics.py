from pathlib import Path
import pandas as pd
import json
from .utils import distance_from_coordinates
 
def classify_result(   
    pred,
    gt,
    is_valid_pred,
    tolerance: float = 5,
    is_good_query: bool = False,
    ) -> str:
    """
    Classify result into TP, FP1, FP2, TN, FN (read docs for meaning).
    Params
    ------ 
    """

    if not is_good_query:
        
        if is_valid_pred:
            return "FP2"
        else:
            return "TN"
    else:
        if not is_valid_pred:
            return "FN"
        else:
            error = distance_from_coordinates(pred, gt)
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
    


    
    
    
    

