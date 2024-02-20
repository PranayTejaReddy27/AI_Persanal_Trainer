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
            self.lmList = []
            if self.results.pose_landmarks:
                for id, lm in enumerate(self.results.pose_landmarks.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x*w), int(lm.y*h)
                    self.lmList.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
            return self.lmList
    def findAngle(self, img,p1, p2, p3, draw = True):

        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        if draw:
            cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)


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
