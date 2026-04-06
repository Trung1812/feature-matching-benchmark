from transformers import AutoImageProcessor, AutoModel
import torch
from PIL import Image
import json
import time
from pathlib import Path
import cv2

def load_uav_metadata(img_path: Path) -> dict:
    """Load metadata JSON that sits next to the UAV image (same stem)."""
    meta_path = img_path.with_suffix(".json")
    if not meta_path.exists():
        return {}
    with open(meta_path, "r", encoding="utf8") as f:
        return json.load(f)

def process(uav_dir: Path, map_dir: Path, result_dir: Path, min_inliers: int=20):
    processor = AutoImageProcessor.from_pretrained("ETH-CVG/lightglue_superpoint")
    model = AutoModel.from_pretrained("ETH-CVG/lightglue_superpoint")

    catalog = {d.name: list(d.glob("*.jpg")) + list(d.glob("*.png"))
               for d in map_dir.iterdir() if d.is_dir()}
    
    if not catalog:
        return

    preprocessing_time = []
    inference_time = []
    postprocessing_time = []

    uav_paths = sorted(p for p in uav_dir.iterdir()
                       if p.suffix.lower() in {".jpg", ".png"})

    for idx, uav_path in enumerate(uav_paths):
        uav_img = Image.open(uav_path)
        for lm, refs in catalog.items():
            for ref in refs:
                lm_img = Image.open(ref)
                images = [uav_img, lm_img]

                inputs = processor(images, return_tensors='pt').to('cuda')
                with torch.inference_mode():
                    outputs = model(**inputs)

                image_sizes = [[(image.height, image.width) for image in images]]

                processed_outputs = processor.post_process_keypoint_matching(outputs, image_sizes, threshold=0.2)

                if len(processed_outputs[0]['matching_scores']) < min_inliers:
                    continue
                
                keypoints1 = processed_outputs[0]['keypoints0'].cpu().numpy()
                keypoints2 = processed_outputs[0]['keypoints1'].cpu().numpy()
                H, mask = cv2.findHomography(keypoints1, keypoints2)
        # free GPU memory
        torch.cuda.empty_cache()
