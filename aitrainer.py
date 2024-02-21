import cv2
import pandas as pd
import numpy as np
import time
import PoseModule as pm


cap = cv2.VideoCapture('Videos/dumbbells.mp4')
detector = pm.poseDetector()
count = 0
dir  = 0
pTime = 0

while True:
    success, img = cap.read()
    #img = cv2.imread('Videos/example.jpg')
    img = cv2.resize(img, (420,840))
    img = detector.findPose(img, False)
    lmList = detector.findPosition(img, False)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    # display FPS
    #cv2.putText(img, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    #print(lmList)
    if len(lmList) != 0: 
        # left arm 
        angle = detector.findAngle(img, 11, 13, 15)
        # right arm
        #detector.findAngle(img, 12, 14, 16)
        per = np.interp(angle,(200,320),(0,100))
        #print(angle,per)
        if per == 100 :
            if dir == 0:
                count += 0.5
                dir = 1

        if per == 0 :
            if dir == 1:
                count += 0.5
                dir = 0
        print(count)
        # draw rectangle 
        cv2.rectangle(img, (20,10),(200,100),(123,232,121),cv2.FILLED)
        # count text display
        cv2.putText(img, str(int(count)), (90,90), cv2.FONT_HERSHEY_SIMPLEX,2, (23,34,10),3)
        # display FPS
    cv2.putText(img,"FPS : " + str(int(fps)), (70,30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)

    cv2.imshow("Image",img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break  # Exit the loop if 'q' is pressed

img.release()  # Release the camera
cv2.destroyAllWindows()
    
