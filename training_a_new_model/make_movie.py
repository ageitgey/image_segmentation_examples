import os
import cv2
import mrcnn.config
import mrcnn.visualize
import mrcnn.utils
from mrcnn.model import MaskRCNN
from pathlib import Path
import numpy as np


# Configuration that will be used by the Mask-RCNN library
class MaskRCNNConfig(mrcnn.config.Config):
    NAME = "coco_pretrained_model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 1  # COCO dataset has 80 classes + one background class


# Root directory of the project
ROOT_DIR = Path(".")

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = "/home/ageitgey/projects/image_segmentation_pyimageconf/training_a_new_model/logs/trashcan20180806T2222/mask_rcnn_trashcan_0030.h5"

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    mrcnn.utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

# Create a Mask-RCNN model in inference mode
model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=MaskRCNNConfig())

# Load pre-trained model
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
class_names = ['BG', 'trashcan']

video_capture = cv2.VideoCapture("trashcan1.mp4")

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(f"out-mask.avi", fourcc, 30.0, (1280, 720))


def visualize_detections(image, masks, scores):
    # Loop over each detected object's mask
    for i in range(masks.shape[2]):
        if scores[i] < 0.981:
            continue

        # Draw the mask for the current object in white
        mask = masks[:, :, i]
        color = (0.0, 0.0, 0.0) # Red
        masked_image = mrcnn.visualize.apply_mask(image, ~mask, color, alpha=1.0)

    return image.astype(np.uint8)

while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break



    # Convert the image from BGR color (which OpenCV uses) to RGB color
    rgb_image = frame[:, :, ::-1]

    # Run the image through the model
    results = model.detect([rgb_image], verbose=1)

    # Visualize results
    r = results[0]
    img = visualize_detections(rgb_image,r['masks'], r['scores'])
    frame = img[:, :, ::-1]

    cv2.imshow('Video', frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
video_capture.release()
cv2.destroyAllWindows()