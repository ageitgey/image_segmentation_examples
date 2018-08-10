import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import mrcnn.config
import mrcnn.visualize
import mrcnn.utils
from mrcnn.model import MaskRCNN
from pathlib import Path


def visualize_detections(image, masks, boxes, class_ids):
    # Create a new solid-black image the same size as the original image
    masked_image = np.zeros(image.shape)

    # Loop over each detected object's mask
    for i in range(masks.shape[2]):
        # If the detected object isn't a person (class_id == 1), skip it
        if class_ids[i] != 1:
            continue

        # Draw the mask for the current object in white
        mask = masks[:, :, i]
        color = (1.0, 1.0, 1.0) # White
        masked_image = mrcnn.visualize.apply_mask(masked_image, mask, color, alpha=1.0)

        # Use Morphological operations (dilate and erode) to find large blobs of people
        kernel = np.ones((5, 5), np.uint8)
        masked_image = cv2.morphologyEx(masked_image, cv2.MORPH_DILATE, kernel, iterations=4)
        masked_image = cv2.morphologyEx(masked_image, cv2.MORPH_ERODE, kernel, iterations=1)

    # Convert the masked image to pure black and white (1-bit)
    image_bw = cv2.cvtColor(masked_image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    thresh, image_bw = cv2.threshold(image_bw, 220, 255, cv2.THRESH_BINARY)

    # Find the single largest contour area which
    (_, cnts, _) = cv2.findContours(image_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]

    # Find the bounding box of the largest counter
    bounding_x1, bounding_y1, bounding_w, bounding_h = cv2.boundingRect(cnts[0])

    # Get the coords of the bottom right corner of the bounding box
    bounding_x2 = bounding_x1 + bounding_w
    bounding_y2 = bounding_y1 + bounding_h

    bgr_image = image[:, :, ::-1]
    font = cv2.FONT_HERSHEY_DUPLEX

    person_count = 0

    # Loop over each detected person
    for i in range(boxes.shape[0]):
        # If the detected object isn't a person (class_id == 1), skip it
        if class_ids[i] != 1:
            continue

        # Get the bounding box of the current person
        y1, x1, y2, x2 = boxes[i]

        # Check if this person is inside the overall line's bounding box
        if x1 >= bounding_x1 and x2 <= bounding_x2 and y1 >= bounding_y1 and y2 <= bounding_y2:
            person_count += 1

           # Draw a mask for the current person
            mask = masks[:, :, i]
            color = (1.0, 1.0, 1.0) # White
            image = mrcnn.visualize.apply_mask(image, mask, color, alpha=0.6)

    # Draw a box around the whole line area
    cv2.rectangle(bgr_image, (bounding_x1, bounding_y1), (bounding_x2, bounding_y2), (0, 0, 255), 14)

    # Label the number of people in line
    cv2.putText(bgr_image, f"{person_count} in line", (bounding_x1, bounding_y1-20), font, 2.0, (0, 0, 255), 5)

    # Convert the image back to RGB
    rgb_image = bgr_image[:, :, ::-1]

    # Return the image and the number of people in line
    return person_count, rgb_image.astype(np.uint8)


# Configuration that will be used by the Mask-RCNN library
class MaskRCNNConfig(mrcnn.config.Config):
    NAME = "coco_pretrained_model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 80  # COCO dataset has 80 classes + one background class


# Root directory of the project
ROOT_DIR = Path(".")

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    mrcnn.utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

# Create a Mask-RCNN model in inference mode
model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=MaskRCNNConfig())

# Load pre-trained model
model.load_weights(COCO_MODEL_PATH, by_name=True)

# Load the image we want to run detection on
image_path = str(ROOT_DIR / "sample_images" / "line_example.png")
image = cv2.imread(image_path)

# Convert the image from BGR color (which OpenCV uses) to RGB color
rgb_image = image[:, :, ::-1]

# Run the image through the model
results = model.detect([rgb_image], verbose=1)

# Visualize results
r = results[0]
people_in_line_count, masked_image = visualize_detections(rgb_image, r['masks'], r['rois'], r['class_ids'])
print(f"Found {people_in_line_count} people in line outside the building")

# Show the result on the screen
plt.imshow(masked_image.astype(np.uint8))
plt.show()
