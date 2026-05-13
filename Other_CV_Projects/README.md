# OpenCV_Python_AI
Examples for fast enabling to typical projects

You need to install following packages to be able to run script:
numpy (1.19.2+)
opencv-python (4.4.0.44)
opencv-contrib-python (4.4.0.44)

#######################################################
#  DocumentScanning_OpenCV.py - OpenCV Document Scanner
#######################################################
A Python-based real-time document scanner that identifies rectangular objects via webcam, corrects their perspective, and provides a flattened "scanned" output.

## Features
* **Live Edge Detection**: Uses Canny edge detection to find document boundaries.
* **Automatic Perspective Correction**: Warps the selected area to a 540x640 rectangle.
* **Visual Workflow**: Displays the original image with contours alongside the warped result.

## Requirements
* Python 3.x
* OpenCV (`opencv-python`)
* NumPy

## How to Use
1. Run the script.
2. Place a high-contrast rectangular object (like a sheet of paper) in front of the camera.
3. The "WorkFlow" window will show the detected contour and the straightened result.
4. Press **'q'** to exit.

#################################################################
# FaceRecognition_OpenCV.py - OpenCV Face Detector (Haar Cascade)
#################################################################

A lightweight Python script for detecting human faces in images using the classical Viola-Jones object detection framework.

## How it Works
1. **Grayscale Conversion**: Reduces data complexity by focusing on pixel intensity.
2. **Haar Features**: Uses rectangular filters to identify facial structures (eyes, nose, mouth).
3. **Cascading Classifier**: A series of stages where the algorithm quickly discards non-face regions.

## Technical Details: Weights & Logic
* **Weights**: In this context, "weights" are stored in the `haarcascade_frontal_default.xml` file. 
* **Architecture**: These are not neural network synapses. Instead, it is a **Decision Tree (Cascading Classifier)**.
* **Origin**: The file contains threshold values for thousands of simple classifiers trained by Intel on a massive database of positive (faces) and negative (non-faces) samples.

## Parameter Tuning (Tips)
If you are experiencing issues with detection accuracy, adjust the `detectMultiScale` parameters:
* **False Positives**: If the script detects boxes where there are no faces, increase `minNeighbors` (e.g., to 3 or 5).
* **Missed Faces**: If the detector fails to see a face, decrease `scaleFactor` closer to **1.1** to make the sliding window search more granular.

## Requirements
* OpenCV (`opencv-python`)
* Pre-trained XML: `haarcascade_frontal_default.xml`

## Usage
Ensure your image path is correct in `cv.imread()`. Adjust `scaleFactor` and `minNeighbors` in `detectMultiScale` to tune detection accuracy versus false positives.

##################################################################
# ShapesRecognition_OpenCV.py - OpenCV Shape Detector & Classifier
##################################################################

A geometric analysis tool that identifies and labels basic shapes (Triangles, Rectangles, Squares, and Circles) within an image.

## Workflow
1. **Preprocessing**: Grayscale conversion and Gaussian Blur to smooth out noise.
2. **Edge Detection**: Canny algorithm defines the boundaries of all objects.
3. **Contour Analysis**: Finds external loops and calculates their area and perimeter.
4. **Polygon Approximation**: Uses the number of vertices to determine the shape type.
5. **Visualization**: Stacks 6 stages of processing into a single view for debugging.

## Features
* Detects shapes based on geometric properties.
* Distinguishes between Squares and Rectangles using aspect ratio logic.
* Filters out noise by ignoring shapes with an area smaller than 250 pixels.

## Requirements
* OpenCV (`opencv-python`)
* NumPy
