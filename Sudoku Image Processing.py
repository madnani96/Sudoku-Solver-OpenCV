from utils import *
import numpy as np
import cv2

from tensorflow.keras.models import load_model
from Sudoku_solver import *
#Comment/Uncomment the path variable to check on different sudoku problems
#path="1.jpg"
#path="2.jpg"
#path = "3.jpg"
#path="4.jpg"
#path="5.jpg"

#Path 6 image 2 blured to process some digits are not classified but still getting a solution from
#whatever digits it is classifying
#path="6.jpg"

#Path 7 image doesnt classify all the
#digits some are left blank as 0 so we get a different solution but that too is correct
#because our backtracking algo is working correctly
#path="7.jpg"

#one misclassified digit in path 8
#path="8.jpg"


#path="9.jpg"
#path="10.jpg"
path="11.jpg"


from Sudoku_solver import *
image = cv2.imread(path)
image = cv2.resize(image, (450, 450))
imgBlank = np.zeros((450, 450, 3), np.uint8)

#Step 1. Preprocessing
#Convert to grayscale
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#Adding Gaussian Blur(Blurring an image to reduce noise and image detail
gaussianblurimage = cv2.GaussianBlur(grayscale_image, (5, 5), 1)
# Applying Adaptive Threshhold
Adaptive_Threshold_Image = cv2.adaptiveThreshold(gaussianblurimage, 255, 1, 1, 11, 2)
#Display Image after Preprocessing
#Uncomment if you want to see image at each step close the image upon viewing for program to run further.
#cv2.imshow("originalimage",image)
#cv2.imshow("Grayscale Image",grayscale_image)
#cv2.imshow("Gaussisan Blur",gaussianblurimage)
#cv2.imshow("Threshhold Image",Adaptive_Threshold_Image)
#cv2.waitKey(0)


#Step 2 Find all contours
contoursimage = image.copy()
contours, h = cv2.findContours(Adaptive_Threshold_Image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(contoursimage, contours, -1, (100, 100, 0), 3)
#Uncomment if you want to see image at each step close the image upon viewing for program to run further.
cv2.imshow("Image Contour",contoursimage)
cv2.waitKey(0)
#print(contours)




#Step3 Extract Biggest Contour
largestcontourimage = image.copy()
largestcontour = np.array([])
maximumarea = 0
for i in contours:
    area = cv2.contourArea(i)
    if area > 50:
        perimeter = cv2.arcLength(i, True)
        approx = cv2.approxPolyDP(i, 0.02 * perimeter, True)
        if area > maximumarea and len(approx) == 4:
            largestcontour = approx
            maximumarea = area


#print("Largest Contour coordinates")
#print(largestcontour)
if largestcontour.size != 0:
    coordinates = largestcontour.reshape((4,2))
    newcoordinates = np.zeros((4,1,2),dtype=np.int32)
    add = coordinates.sum(1)
    newcoordinates[0] = coordinates[np.argmin(add)]
    newcoordinates[3] = coordinates[np.argmax(add)]
    difference = np.diff(coordinates,axis=1)
    newcoordinates[1] = coordinates[np.argmin(difference)]
    newcoordinates[2] = coordinates[np.argmax(difference)]
    largestcontour = newcoordinates
    #print(largestcontour)

    #Drawing the biggest contour
    cv2.drawContours(largestcontourimage,largestcontour,-1,(0,100,200),25)
    #wrap perspective transformation
    p1 = np.float32(largestcontour)
    p2 = np.float32([[0,0],[450,0],[0,450],[450,450]])
    perspectiveinputmatrix=cv2.getPerspectiveTransform(p1,p2)
    WarpColoredimage=cv2.warpPerspective(image,perspectiveinputmatrix,(450,450))
    cv2.imshow("Largest Contour Image",largestcontourimage)
    cv2.waitKey(0)
    WarpColoredimage = cv2.warpPerspective(image, perspectiveinputmatrix, (450, 450))
    WarpColoredimagetograyscale=cv2.cvtColor(WarpColoredimage, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Warp Colored Image", WarpColoredimagetograyscale)
    cv2.waitKey(0)


#Step 4
rows = np.vsplit(WarpColoredimagetograyscale,9)
boxes=[]
#boxes will store 81 different images
for r in rows:
    columns= np.hsplit(r,9)
    for box in columns:
        boxes.append(box)
#print(len(boxes))
cv2.imshow("sample box",boxes[1])
cv2.waitKey(0)


model = load_model('Myfirstmodel/')

result = []
for image in boxes:
    testingimg = np.asarray(image)
    testingimg = testingimg[4:(testingimg.shape[0]- 4), 4:(testingimg.shape[1] - 4)]
    testingimg = (cv2.resize(testingimg, (32,32)))/255

    testingimg = testingimg.reshape(1,32,32,1)
    predictions = model.predict(testingimg)
    cindex = np.argmax(predictions,axis=-1)
    probability = np.amax(predictions)
    if probability > 0.75:
        result.append(cindex[0])
    else:
        result.append(0)

print("Sudoku problem extracted from the image")
print(result)

res = np.reshape(result,(9,9))
print("Sudoku Problem Solution:")
sudoku(res)


#print(numbers)




