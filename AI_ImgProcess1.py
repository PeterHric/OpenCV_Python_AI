# (c) Hassan Murtaza : https://github.com/murtazahassan/Learn-OpenCV-in-3-hours/blob/master/chapter8.py

import cv2 as cv
import numpy as np

# Stacks images to the row, with given scale
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv.cvtColor( imgArray[x][y], cv.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv.cvtColor(imgArray[x], cv.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

# Draws contours in the given image
def getContours(in_img, draw_to_img):

    # contours, hierarchy = cv.findContours(imgBlur, cv.Mode, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # cv.RETR_EXTERNAL is specifically good for retrieving the outer boundaries of object (perhaps great for image recognition)
    # Why cv.CHAIN_APPROX_NONE ? Investigate and try others..
    contours, hierarchy = cv.findContours(in_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    for cnt in contours:
        area = cv.contourArea(cnt, None)  # oriented = None
        print(area)
        cv.drawContours(draw_to_img, cnt, -1, (255, 0, 0), 3)  # -1 to draw all contours


filePath = "Resources/lena_Large.jpg"
# img = cv.imread(fileName, -1)
img = cv.imread(filePath)

# print(img)
# cv.imshow('Original', img)

# Add text
text = 'Lena width: ' + str(img.shape[0]) + ' height: ' + str(img.shape[1])
cv.putText(img, text, (25, 25), cv.FONT_HERSHEY_SIMPLEX, 1, (230, 230, 25), 2)

# Convert to BW
imgBW = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# Add blur for better transition (EDGE) detection
sigmaX_gauss = 1
sigmaY_gauss = 1
conv_kernel_mtx = (5, 5)  # have to be odd numbers !
imgBlur = cv.GaussianBlur(imgBW, conv_kernel_mtx, sigmaX_gauss, sigmaY_gauss)  # Y sigma defaulted to 1

# Find contours
imgCanny = cv.Canny(imgBlur, 50, 50)

# Blank helper
imgBlank = np.zeros_like(img)

imgContour = img.copy()  # Something like a (deep?) copy c-tor
# imgContour = np.array(img).reshape((-1,1,2)).astype(np.int32)
getContours(imgCanny, imgContour)

imgStack = stackImages(0.6, ([img, imgBW, imgBlur],
                             [imgCanny, imgContour, imgBlank]))

#imgStack = np.vstack((np.hstack((img, imgBW, imgBlur)),
#                      np.hstack((imgCanny, imgContour, imgBlank))))

cv.imshow("Orig, BW, Blur, Blank", imgStack)

cv.waitKey(0)

# ToDo: Recognise the image - face, shape, animal, fruit, tool, ....
