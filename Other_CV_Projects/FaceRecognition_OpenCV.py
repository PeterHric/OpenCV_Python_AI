# Face detection script using the Viola-Jones framework (Haar Cascades).
# 1. Load pre-trained Haar Cascade XML classifier for frontal faces.
# 2. Read input image and convert it to grayscale (required for the classifier).
# 3. Multiscale detection: Sliding window approach to find faces of various sizes.
# 4. Visualization: Draw bounding boxes (rectangles) around detected coordinates.

import cv2 as cv

# Load the pre-trained Haar Cascade model from an XML file
faceCascade = cv.CascadeClassifier("Resources/haarcascade_frontal_default.xml")

# img = cv.imread('Resources/lena_Large.jpg')
img = cv.imread('Resources/Face_1.jpeg')

# Convert to grayscale as Haar cascades operate on intensity rather than color
imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Detect and store the faces into an object describing the rectangle around detected face
# detectMultiScale searches for objects of different sizes in the input image
faces = faceCascade.detectMultiScale(imgGray,
                                     1.4,  # Scale factor: How much the image size is reduced at each scale
                                     1)    # Min neighbours: How many neighbors each candidate rectangle should have to retain it

# Iterate through all detected face coordinates and draw blue rectangles
for (x, y, w, h) in faces:
    cv.rectangle(img, (x, y),(x+w, y+h), (255, 0 ,0), 2)

# Display the final image with detected faces
cv.imshow("Result", img)
cv.waitKey(0)
