import os
import cv2
import mrcnn.config
import mrcnn.visualize
import mrcnn.utils
from mrcnn.model import MaskRCNN
from pathlib import Path


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

# Load the image we want to run detection on
image_path = "/home/ageitgey/projects/image_segmentation_pyimageconf/training_a_new_model/SD Trash Pics/Photo Jul 30, 12 40 33 PM.jpg"
image = cv2.imread(image_path)

# Convert the image from BGR color (which OpenCV uses) to RGB color
rgb_image = image[:, :, ::-1]

# Run the image through the model
results = model.detect([rgb_image], verbose=1)

# Visualize results
r = results[0]
mrcnn.visualize.display_instances(rgb_image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
