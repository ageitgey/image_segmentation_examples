## Installation

### Install Python

I recommend installing Python 3.6. I typically use the installers from http://python.org.

### Install TensorFlow with GPU support

Follow the instructions on their website to install TensorFlow for Python 3.6: 
https://www.tensorflow.org/install/

It will still work if you don't have a GPU and install the non-GPU version, but 
it will be much slower.

### Install OpenCV 3

Install OpenCV 3 following the installation guides on http://pyimagesearch.com.

Make sure you install it with Python bindings for the copy of Python 3.6 you are using.

For example, the Mac OpenCV 3 / Python 3 install guide is here: 

https://www.pyimagesearch.com/2016/12/05/macos-install-opencv-3-and-python-3-5/

### Install mask-rcnn

Git clone the source from https://github.com/matterport/Mask_RCNN:

```bash
git clone https://github.com/matterport/Mask_RCNN.git 
```

Install required libraries:

```bash
pip3 install -r requirements.txt
```

Install the COCO python api:

Unfortunately this is a bit difficult as it isn't well maintained. You have to install a fork for your OS:

 - Linux: https://github.com/waleedka/coco
 - Windows: https://github.com/philferriere/cocoapi. You must have the Visual C++ 2015 build tools on your path (see the repo for additional details)

Install mask-rcnn so it's available to other Python programs:

```bash
python3.6 setup.py install
```

Congrats, you are ready to run the code in this repo!