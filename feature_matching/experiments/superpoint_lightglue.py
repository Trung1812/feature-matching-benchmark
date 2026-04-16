import cv2
from transformers import AutoImageProcessor, AutoModel
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from feature_matching.experiments.utils import *
from loguru import logger

def get_center_loc(h_uav_to_sat, h_sat_to_utm, translation, height, width):
    #TODO: You have to transform them to float64
    center_x = height / 2
    center_y = width / 2
    xy = np.array([[[center_x, center_y]]], dtype=np.float64)

    homography = translation @ h_sat_to_utm @ h_uav_to_sat

    transformed_x = cv2.perspectiveTransform(xy, homography)

    return transformed_x

def process_one_pair(processor, model, 
                      uav_img, lm_img, uav_path, ref_path,
                      lm_homography, lm_translation,
                      result_dir,
                      min_inliers, min_inliers_vis=10):
    """
    
    Output 
    """
    images = [uav_img, lm_img]
    inputs = processor(images, return_tensors='pt').to('cuda')

    with torch.inference_mode():
        outputs = model(**inputs)

    image_sizes = [[(image.height, image.width) for image in images]]

    processed_outputs = processor.post_process_keypoint_matching(outputs, image_sizes, threshold=0.2)

    if len(processed_outputs[0]['matching_scores']) < min_inliers:
        return None
    
    keypoints1 = processed_outputs[0]['keypoints0'].cpu().numpy()
    keypoints2 = processed_outputs[0]['keypoints1'].cpu().numpy()
    H, mask = cv2.findHomography(keypoints1, keypoints2, method=cv2.USAC_MAGSAC)

    total_matches = len(mask)
    if mask is not None:
        num_inliers = mask.sum()
    else:
        num_inliers = 0
    
    predicted_coordinate = get_center_loc(H, lm_homography, lm_translation, uav_img.height, uav_img.width)

    
    if num_inliers >= min_inliers_vis:
        vis = processor.visualize_keypoint_matching(images, processed_outputs)

        # Save image to result_dir
        vis_name = f"{uav_path.stem}+{ref_path.stem}.jpg"
        vis_path = result_dir / vis_name
        vis[0].save(vis_path)

    # predicted_keypoints_uav_coordinates = ...
    return predicted_coordinate, H, num_inliers, total_matches, keypoints1, keypoints2

def process_one_query():
    pass

def process(uav_dir: Path,
            landmark_dir: Path,
            result_dir: Path,
            min_inliers: int=20,
            min_inliers_vis: int=10):

    result_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoImageProcessor.from_pretrained("ETH-CVG/lightglue_superpoint")
    model = AutoModel.from_pretrained("ETH-CVG/lightglue_superpoint")
    model.to('cuda')

    catalog = sorted(list(landmark_dir.glob("*.jpg")))
    
    if not catalog:
        return

    cover = get_cover(landmark_dir)

    uav_paths = sorted(list(uav_dir.glob("*.jpg")))

    for idx, uav_path in enumerate(uav_paths):
        logger.info(f"Processing query {idx+1}/{len(uav_paths)}: {uav_path.name}")
        uav_img = Image.open(uav_path)
        uav_center = get_uav_center(uav_path)
        logger.info(f"UAV center coordinates: {uav_center}")
        x, y = uav_center
        uav_is_good_query = is_good_query(x, y, cover)

        for ref in catalog:
            lm_img = Image.open(ref)
            lm_homography, lm_translation = get_landmark_translation_and_homography(ref)
            result = process_one_pair(processor, model, uav_img, lm_img, uav_path, ref, lm_homography, lm_translation, result_dir, min_inliers, min_inliers_vis)
            if result is not None:
                predicted_coordinate, h_uav_to_sat, num_inliers, total_matches, keypoints_uav, keypoints_sat = result
                # save results
                result = {
                    "is_good_query": uav_is_good_query,
                    "ref_path": str(ref),
                    "predicted_coordinate": predicted_coordinate.tolist(),
                    "num_inliers": int(num_inliers),
                    "total_matches": int(total_matches),
                    "keypoints_uav": keypoints_uav.tolist(),
                    "keypoints_sat": keypoints_sat.tolist(),
                    "h_uav_to_sat": h_uav_to_sat.tolist()
                }
                if num_inliers > num_inliers:
                    save_result(result_dir, uav_path, result)
            else:
                continue
        # free GPU memory
        torch.cuda.empty_cache()

if __name__=="__main__":
    uav_dir = "/home/c4i/workspace/trungpq16/uav220/feature_matching_experiments/data/processed/queries/uav10"
    landmark_dir = "/home/c4i/workspace/trungpq16/uav220/feature_matching_experiments/data/processed/landmarks/25K_test_set" 
    result_dir = "/home/c4i/workspace/trungpq16/uav220/feature_matching_experiments/reports/experiment_results/superpoint_lightglue/hoalac_full_default_config"
    uav_dir = Path(uav_dir)
    landmark_dir = Path(landmark_dir)
    result_dir = Path(result_dir)
    process(uav_dir, landmark_dir, result_dir)