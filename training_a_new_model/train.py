# This demo is roughly based on the balloon.py demo included with Matterport's Mask R-CNN implementation which
# is licensed under the MIT License (see LICENSE for details).
# This is based on code originally written by Waleed Abdulla but re-written to support data annotated by RectLabel.

import warnings
from pathlib import Path
import xmltodict
import numpy as np
import skimage.draw
from mrcnn.config import Config
from mrcnn import model as modellib, utils

ROOT_DIR = Path(".")
COCO_WEIGHTS_PATH = ROOT_DIR / "mask_rcnn_coco.h5"

# Download COCO trained weights if you don't already have them.
if not COCO_WEIGHTS_PATH.exists():
    utils.download_trained_weights(str(COCO_WEIGHTS_PATH))

# Directory to save logs and model checkpoints
DEFAULT_LOGS_DIR = ROOT_DIR / "training_logs"

# Where the training images and annotation files live
DATASET_PATH = ROOT_DIR / "training_images"

# Start training from the pre-trained COCO model. You can change this path if you want to pick up training from a prior
# checkpoint file in your ./logs folder.
WEIGHTS_TO_START_FROM = COCO_WEIGHTS_PATH


class ObjectDetectorConfig(Config):
    NAME = "custom_object"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # Background + your custom object
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


class RectLabelDataset(utils.Dataset):

    def load_training_images(self, dataset_dir, subset):
        dataset_dir = dataset_dir / subset
        annotation_dir = dataset_dir / "annotations"

        # Add classes. We have only one class to add since this model only detects one kind of object.
        self.add_class("custom_object", 1, "custom_object")

        # Load each image by finding all the RectLabel annotation files and working backwards to the image.
        # This is a lot faster then having to load each image into memory.
        count = 0

        for annotation_file in annotation_dir.glob("*.xml"):
            print(f"Parsing annotation: {annotation_file}")
            xml_text = annotation_file.read_text()
            annotation = xmltodict.parse(xml_text)['annotation']
            objects = annotation['object']
            image_filename = annotation['filename']
            if not isinstance(objects, list):
                objects = [objects]

            # Add the image to the data set
            self.add_image(
                source="custom_object",
                image_id=count,
                path=dataset_dir / image_filename,
                objects=objects,
                width=int(annotation["size"]['width']),
                height=int(annotation["size"]['height']),
            )
            count += 1


    def load_mask(self, image_id):
        # We have to generate our own bitmap masks from the RectLabel polygons.

        # Look up the current image id
        info = self.image_info[image_id]

        # Create a blank mask the same size as the image with as many depth channels as there are
        # annotations for this image.
        mask = np.zeros([info["height"], info["width"], len(info["objects"])], dtype=np.uint8)

        # Loop over each annotation for this image. Each annotation will get it's own channel in the mask image.
        for i, o in enumerate(info["objects"]):
            # RectLabel uses Pascal VOC format which is kind of wacky.
            # We need to parse out the x/y coordinates of each point that make up the current polygon
            ys = []
            xs = []
            for label, number in o["polygon"].items():
                number = int(number)
                if label.startswith("x"):
                    xs.append(number)
                else:
                    ys.append(number)

            # Draw the filled polygon on top of the mask image in the correct channel
            rr, cc = skimage.draw.polygon(ys, xs)
            mask[rr, cc, i] = 1

        # Return mask and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        # Get the path for the image
        info = self.image_info[image_id]
        return info["path"]


def train(model):
    # Load the training data set
    dataset_train = RectLabelDataset()
    dataset_train.load_training_images(DATASET_PATH, "training_set")
    dataset_train.prepare()

    # Load the validation data set
    dataset_val = RectLabelDataset()
    dataset_val.load_training_images(DATASET_PATH, "validation_set")
    dataset_val.prepare()

    with warnings.catch_warnings():
        # Suppress annoying skimage warning due to code inside Mask R-CNN.
        # Not needed, but makes the output easier to read until Mask R-CNN is updated.
        warnings.simplefilter("ignore")

        # Re-train the model on a small data set. If you are training from scratch with a huge data set,
        # you'd want to train longer and customize these settings.
        model.train(
            dataset_train,
            dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=30,
            layers='heads'
        )


# Load config
config = ObjectDetectorConfig()
config.display()

# Create the model
model = modellib.MaskRCNN(mode="training", config=config, model_dir=DEFAULT_LOGS_DIR)

# Load the weights we are going to start with
# Note: If you are picking up from a training checkpoint instead of the COCO weights, remove the excluded layers.
model.load_weights(str(WEIGHTS_TO_START_FROM), by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

# Run the training process
train(model)
