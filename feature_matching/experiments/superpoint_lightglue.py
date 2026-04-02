from transformers import AutoImageProcessor, SuperPointForKeypointDetection
import torch
from PIL import Image

url = "/home/c4i/Documents/uav2-30/map/1/1.jpg"
image = Image.open(url)

processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")
model = SuperPointForKeypointDetection.from_pretrained("magic-leap-community/superpoint")

inputs = processor(image, return_tensors='pt')
with torch.no_grad():
    outputs = model(**inputs)

import matplotlib.pyplot as plt

plt.axis("off")
plt.imshow(image)
plt.scatter(
    outputs["keypoints"][:, 0],
    outputs["keypoints"][:, 1],
    c=outputs["scores"],
    s=outputs["scores"] * 50,
    alpha=0.8
)

plt.show()