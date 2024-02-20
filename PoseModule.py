import cv2
import mediapipe as mp
import time 

class poseDetector():

    def __init__(self, mode=False, modelcom=1, smooth=True, enableseg = False,
                  smoothseg = True, detectioncon=0.5, trackcon=0.5):
        self.mode = mode
        self.modelcom = modelcom
        self.smooth = smooth
        self.enableseg = enableseg
        self.smoothseg = smoothseg
        self.detectioncon = detectioncon
        self.trackcon = trackcon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.modelcom, self.smooth, self.enableseg, self.smoothseg, self.detectioncon, self.trackcon)

    def findPose(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        #print(results.pose_landmarks)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img
    
    def findPosition(self, img, draw = True):
            lmList = []
            if self.results.pose_landmarks:
                for id, lm in enumerate(self.results.pose_landmarks.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x*w), int(lm.y*h)
                    lmList.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
            return lmList
        

def main():
    cap = cv2.VideoCapture('Videos/dumbbells.mp4')

    pTime = 0
    detector = poseDetector()
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img)
        print(lmList[12])
        cv2.circle(img, (lmList[12][1], lmList[12][2]), 25, (0, 0, 255), cv2.FILLED)
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        img = cv2.resize(img, (420,820))
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
