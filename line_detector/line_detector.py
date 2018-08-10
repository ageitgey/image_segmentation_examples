import os
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
import mrcnn.config
import mrcnn.visualize
import mrcnn.utils
from mrcnn.model import MaskRCNN


def draw_line_length(image, boxes, masks, class_ids, class_names,
                     scores=None, title="",
                     figsize=(16, 16), ax=None,
                     show_mask=True, show_bbox=True,
                     colors=None, captions=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = colors or mrcnn.visualize.random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = np.zeros(image.shape)
    color_image = image.astype(np.uint8).copy()

    for i in range(N):
        color = (1.0, 1.0, 1.0)

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        # if show_bbox:
        #     p = visualize.patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
        #                         alpha=0.7, linestyle="dashed",
        #                         edgecolor=color, facecolor='none')
        #     ax.add_patch(p)

        class_id = class_ids[i]
        if class_id != 1:
            continue

        score = scores[i] if scores is not None else None
        label = class_names[class_id]
        x = random.randint(x1, (x1 + x2) // 2)
        caption = "{} {:.3f}".format(label, score) if score else label

        # ax.text(x1, y1 + 8, caption,
        #         color='w', size=11, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = mrcnn.visualize.apply_mask(masked_image, mask, color, alpha=1.0)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = mrcnn.visualize.find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = mrcnn.visualize.Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)

    kernel = np.ones((5, 5), np.uint8)
    masked_image = cv2.morphologyEx(masked_image, cv2.MORPH_DILATE, kernel, iterations=10)
    masked_image = cv2.morphologyEx(masked_image, cv2.MORPH_ERODE, kernel, iterations=4)

    im_bw = cv2.cvtColor(masked_image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    print(im_bw.shape)
    thresh, im_bw = cv2.threshold(im_bw, 220, 255, cv2.THRESH_BINARY)
    print(im_bw.shape)
    print(im_bw.dtype)
    # (thresh, im_bw) = cv2.threshold(im_bw, tresh_min, tresh_max, 0)
    (_, cnts, _) = cv2.findContours(im_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]

    x, y, w, h = cv2.boundingRect(cnts[0])
    h = 16
    x -= 6
    # cv2.drawContours(color_image, [cnts[0]], -1, (255, 255, 255, 0.3), cv2.FILLED)

    cv2.rectangle(color_image, (x, y+h-4), (x + w, y + h), (0, 0, 255), cv2.FILLED)

    cv2.rectangle(color_image, (x, y), (x + 4, y + h), (0, 0, 255), cv2.FILLED)
    cv2.rectangle(color_image, (x+w-4, y), (x + w, y + h), (0, 0, 255), cv2.FILLED)

    font = cv2.FONT_HERSHEY_DUPLEX
    wait = int((w / 16) * 2)
    cv2.putText(color_image, f"Wait - ~{wait} minutes", (x + 5, y + h - 8), font, 1.0, (255, 255, 255), 1)




    plt.imshow(color_image.astype(np.uint8))
    # cv2.imsave("mask.png", masked_image.astype(np.uint8))
    if auto_show:
        plt.show()


# Configuration that will be used by the Mask-RCNN library
class MaskRCNNConfig(mrcnn.config.Config):
    NAME = "coco_pretrained_model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 80  # COCO dataset has 80 classes + one background class


# Root directory of the project
ROOT_DIR = os.path.abspath(".")

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

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


# video_capture = cv2.VideoCapture(0)

while True:
    # Grab a single frame of video
    # ret, frame = video_capture.read()
    # frame = cv2.imread("/Users/ageitgey/Desktop/Screen Shot 2018-07-18 at 10.11.49 AM.png")
    frame = cv2.imread("/Users/ageitgey/Desktop/Screen Shot 2018-07-24 at 4.09.36 PM.png")
    frame = cv2.imread("/Users/ageitgey/Dropbox (Novel-T Sarl)/Novel-T - eBooks and Training/Tech Sessions/2017-11-01 - Machine Learning/example_code/02_neural_networks_keras_and_tensorflow/bay.jpg")

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color
    rgb_small_frame = small_frame[:, :, ::-1]


    results = model.detect([rgb_small_frame], verbose=1)

    # Visualize results
    r = results[0]
    draw_line_length(rgb_small_frame, r['rois'], r['masks'], r['class_ids'],
                     class_names, r['scores'])

    # print(r['rois'])
    # print(r['masks'])
    # print(r['class_ids'])
    # print(r['scores'])
    print(results)

    quit()