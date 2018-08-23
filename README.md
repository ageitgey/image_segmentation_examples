# Image Segmentation examples with Mask R-CNN.

This is the example code for my talk at PyImageConf 2018 on Image Segmentation.

This code uses the [Mask R-CNN implementation from Matterport](https://github.com/matterport/Mask_RCNN).

These examples are roughly inspired by the examples included with that project, so check those out too!


## Installation

### Recommended Hardware

- A computer with an nvidia GT 980 Ti+ or GT 1080 Ti GPU. 
  - You need a a GPU with least 6GB RAM to run the training script. It will fail with less RAM.

If you don't have a GPU, the scripts will still run - just very slowly.

### Install Python 3.6

I recommend installing Python 3.6.6. 

I use the installers from http://python.org for Mac and Windows. I don't use Anaconda.

### Install TensorFlow with GPU support

Follow the instructions on the TensorFlow website to install TensorFlow with GPU support
for Python 3.6: 

https://www.tensorflow.org/install/

This code will still work if you don't have a GPU and install the non-GPU version of 
TensorFlow, but it will run much more slowly. The training example will be especially painful.

### Install OpenCV 3

Install OpenCV 3 following the installation guides on http://pyimagesearch.com.

Make sure you install it with Python bindings for the copy of Python 3.6 you are using.

For example, the Mac OpenCV 3 / Python 3 install guide is here: 

https://www.pyimagesearch.com/2016/12/05/macos-install-opencv-3-and-python-3-5/

### Install the Mask R-CNN implementation

Unfortunately Matterport's Mask R-CNN implementation isn't currently installable from pip, so you have 
to download and install it manually. Here are the steps:

Step 1: Git clone the source from https://github.com/matterport/Mask_RCNN:

```bash
git clone https://github.com/matterport/Mask_RCNN.git 
```

Step 2: Install it's required libraries:

```bash
cd Mask_RCNN
pip3 install -r requirements.txt
```

Step 3: Install the COCO python API that it also requires:

Unfortunately this step is a bit difficult as it isn't well maintained. You have to install a fork for your OS:

 - Linux / Mac: https://github.com/waleedka/coco
 - Windows: https://github.com/philferriere/cocoapi. You must have the Visual C++ 2015 build tools on your path
  (see the repo for additional details)

For me, the steps were:

```bash
git clone https://github.com/waleedka/coco
cd coco/PythonAPI
make
sudo make install
```

Step 4: Finally, install the Mask R-CNN library itself globally so it's available to other Python programs:

```bash
python3.6 setup.py install
```

Congrats, you are now ready to run the code in this repo!

## Included Examples

### Line Detector

The first example shows a simple approach for building a program that detects how many people are waiting in a line.

#### 01_basic_detection.py

Downloads and runs the pre-trained COCO image segmentation model against an image.

#### 02_generate_person_masks.py

This creates a 1-bit mask of areas that contain people in an image.

#### 03_apply_morphological_operations.py

Clumps together groups of people using OpenCV's morpological operations.

#### 04_find_contours.py

Finds the largest clump of people using OpenCV's findContours.

#### 05_count_people_in_line.pu

Counts the number of people waiting in line by seeing which detected people are inside the detected line area.


### Training a New Model

This example shows how to re-train the COCO model against a new dataset using transfer learning.

To do this, you'll need to gather training images, put them in the "training_images" subfolders and annotation
those images with [RectLabel](https://rectlabel.com/). The training images I showed in the talk aren't included,
but I did include one annotated training image and one annotated validation image of a cup just to show you 
the idea and the file formats.

#### train.py

Uses transfer learning to re-train the COCO model to detect a custom object. Uses the training data in the
./training_images folder. 

If you want to build a your own custom model, you can supply your own training images and annotate those images with 
RectLabel. Make sure you include and annotate training images and validation images.

#### evaluate_model.py

Runs your new model on a test image and displays the detections on the screen. You'll want to update
the file path to your trained model and update the path to the image you want to test.

#### evaluate_video_with_model.py

Runs your new model on a video file and displays the detections on the screen and writes out a new
video with the detections. You'll want to update the file path to your trained model and update the 
path to the video file that you want to process.