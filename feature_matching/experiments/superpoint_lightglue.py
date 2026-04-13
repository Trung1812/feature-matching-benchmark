import cv2
from transformers import AutoImageProcessor, AutoModel
import torch
import numpy as np
from PIL import Image
import json
import time
from pathlib import Path

def load_uav_metadata(img_path: Path) -> dict:
    """Load metadata JSON that sits next to the UAV image (same stem)."""
    meta_path = img_path.with_suffix(".json")
    if not meta_path.exists():
        return {}
    with open(meta_path, "r", encoding="utf8") as f:
        return json.load(f)

def process(uav_dir: Path, landmark_dir: Path, result_dir: Path, min_inliers: int=20):
    result_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoImageProcessor.from_pretrained("ETH-CVG/lightglue_superpoint")
    model = AutoModel.from_pretrained("ETH-CVG/lightglue_superpoint")
    model.to('cuda')

    catalog = list(landmark_dir.glob("*.jpg"))
    
    if not catalog:
        return

    preprocessing_time = []
    inference_time = []
    postprocessing_time = []
    homography_calculation = []
    total = []

    uav_paths = sorted(list(uav_dir.glob("*.jpg")))

    for idx, uav_path in enumerate(uav_paths):
        uav_img = Image.open(uav_path)
        for ref in catalog:
                preprocess_start = time.perf_counter()
                lm_img = Image.open(ref)
                images = [uav_img, lm_img]

                inputs = processor(images, return_tensors='pt').to('cuda')
                preprocessing_time.append(time.perf_counter() - preprocess_start)

            
                inference_start = time.perf_counter()
                with torch.inference_mode():
                    outputs = model(**inputs)
                inference_time.append(time.perf_counter() - inference_start)

                image_sizes = [[(image.height, image.width) for image in images]]


                postprocess_start = time.perf_counter()
                processed_outputs = processor.post_process_keypoint_matching(outputs, image_sizes, threshold=0.2)
                postprocessing_time.append(time.perf_counter() - postprocess_start)
                if len(processed_outputs[0]['matching_scores']) < min_inliers:
                    continue
                

                homography_start = time.perf_counter()
                keypoints1 = processed_outputs[0]['keypoints0'].cpu().numpy()
                keypoints2 = processed_outputs[0]['keypoints1'].cpu().numpy()
                H, mask = cv2.findHomography(keypoints1, keypoints2, method=cv2.USAC_MAGSAC)
                homography_calculation.append(time.perf_counter() - homography_start)
                total.append(time.perf_counter() - preprocess_start)

                if mask is not None:
                    num_inliers = mask.sum()
                else:
                    num_inliers = 0
                
                print(f"NUM INLIER: {num_inliers}")
                
                if num_inliers >= min_inliers:
                    vis = processor.visualize_keypoint_matching(images, processed_outputs)
            
                    # Save image to result_dir
                    vis_name = f"{uav_path.stem}+{ref.stem}.jpg"
                    vis_path = result_dir / vis_name
                    vis[0].save(vis_path)
        # free GPU memory
        torch.cuda.empty_cache()

    def summarize(arr: np.ndarray) -> dict:
        if arr.size == 0:
            return {
                "count": 0,
                "min": np.nan,
                "max": np.nan,
                "p90": np.nan,
                "p95": np.nan,
                "p99": np.nan,
                "mean": np.nan,
                "median": np.nan,
            }
        return {
            "count": int(arr.size),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
        }

    timing_arrays = {
        "preprocessing": np.array(preprocessing_time, dtype=np.float64),
        "inference": np.array(inference_time, dtype=np.float64),
        "postprocessing": np.array(postprocessing_time, dtype=np.float64),
        "homography_calculation": np.array(homography_calculation, dtype=np.float64),
        "total": np.array(total, dtype=np.float64)
    }

    report_path = result_dir / "timing_reports.txt"
    with open(report_path, "w", encoding="utf8") as f:
        for stage_name, values in timing_arrays.items():
            stats = summarize(values)
            f.write(f"[{stage_name}]\n")
            f.write(f"count: {stats['count']}\n")
            f.write(f"min: {stats['min']}\n")
            f.write(f"max: {stats['max']}\n")
            f.write(f"p90: {stats['p90']}\n")
            f.write(f"p95: {stats['p95']}\n")
            f.write(f"p99: {stats['p99']}\n")
            f.write(f"mean: {stats['mean']}\n")
            f.write(f"median: {stats['median']}\n\n")

if __name__=="__main__":
    uav_dir = "/home/c4i/workspace/trungpq16/uav220/feature_matching_experiments/data/processed/queries/uav10"
    landmark_dir = "/home/c4i/workspace/trungpq16/uav220/feature_matching_experiments/data/processed/landmarks/landmark_hoalac_jpg" 
    result_dir = "/home/c4i/workspace/trungpq16/uav220/feature_matching_experiments/reports/experiment_results/superpoint_lightglue/hoalac_default_config"
    uav_dir = Path(uav_dir)
    landmark_dir = Path(landmark_dir)
    result_dir = Path(result_dir)
    process(uav_dir, landmark_dir, result_dir)