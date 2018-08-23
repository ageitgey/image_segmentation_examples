import cv2
import mrcnn.config
import mrcnn.visualize
import mrcnn.utils
from mrcnn.model import MaskRCNN
from pathlib import Path


# Configuration that will be used by the Mask-RCNN library
class ObjectDetectorConfig(mrcnn.config.Config):
    NAME = "custom_object"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 1  # custom object + background class


def mask_image(image, masks):
    # Loop over each detected object's mask
    for i in range(masks.shape[2]):
        # Draw the mask for the current object
        mask = masks[:, :, i]
        color = (1.0, 0.0, 0.0) # Red
        image = mrcnn.visualize.apply_mask(image, mask, color, alpha=0.5)

    return image

# Root directory of the project
ROOT_DIR = Path(".")
MODEL_DIR = ROOT_DIR / "training_logs"

# Local path to trained weights file (make sure you update this)
TRAINED_MODEL_PATH = MODEL_DIR / "custom_object20180821T1551" / "mask_rcnn_custom_object_0030.h5"

# Video file to process (make sure you update this too!)
SOURCE_VIDEO_FILE = "your_video_file.mp4"

# Create a Mask-RCNN model in inference mode
model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=ObjectDetectorConfig())

# Load pre-trained model
model.load_weights(str(TRAINED_MODEL_PATH), by_name=True)

# COCO Class names
class_names = ['BG', 'custom_object']

# Load the video file we want to run detection on
video_capture = cv2.VideoCapture(SOURCE_VIDEO_FILE)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(f"{SOURCE_VIDEO_FILE}_detected.avi", fourcc, 30.0, (1920, 1080))

# NOTE:
# This is just a simple demo to show how to process a video, not production quality.
# You could make this run faster by processing more than one image at a time though the model.
# To do that, you need to edit ObjectDetectorConfig to increase "Images per GPU" and then
# pass in batches of frames instead of single frames.
# However, that would require a GPU with more RAM so it may not work for you if you don't
# have a high-end GPU with 12gb of RAM.
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
    rgb_image = mask_image(rgb_image, r['masks'])

    # Convert the image back to BGR
    bgr_image = rgb_image[:, :, ::-1]

    cv2.imshow('Video', bgr_image)
    out.write(bgr_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
video_capture.release()
cv2.destroyAllWindows()