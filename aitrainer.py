import cv2
import pandas as pd
import time

cap = cv2.VideoCapture('Videos/dumbbells.mp4')

while True:
    success, img = cap.read()
    #img = cv2.resize(img, (1280,720))
    cv2.imshow("Image",img)
    cv2.waitKey(1)
    
