import cv2 as cv

faceCascade = cv.CascadeClassifier("Resources/haarcascade_frontal_default.xml")
# img = cv.imread('Resources/lena_Large.jpg')
img = cv.imread('Resources/Face_1.jpeg')
imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Detect and store the faces into an object describing the rectangle around detected face
faces = faceCascade.detectMultiScale(imgGray,
                                     1.4,  #  Scale factor  - try tweak
                                     1)    #  Min neighbours - try tweak

for (x, y, w, h) in faces:
    cv.rectangle(img, (x, y),(x+w, y+h), (255, 0 ,0), 2)

cv.imshow("Result", img)
cv.waitKey(0)