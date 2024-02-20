import cv2
import pandas as pd
import time
import PoseModule as pm


#cap = cv2.VideoCapture('Videos/dumbbells.mp4')
detector = pm.poseDetector()

while True:
    #success, img = cap.read()
    img = cv2.imread('Videos/example.jpg')
    #img = cv2.resize(img, (1280,720))
    img = detector.findPose(img)
    lmList = detector.findPosition(img, False)
    #print(lmList)
    if len(lmList) != 0:
        detector.findAngle(img, 11, 13, 15)


    cv2.imshow("Image",img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break  # Exit the loop if 'q' is pressed

img.release()  # Release the camera
cv2.destroyAllWindows()
    
