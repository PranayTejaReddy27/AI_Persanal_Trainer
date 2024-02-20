from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import time 

app = Flask(__name__)

class poseDetector():
    def __init__(self, mode=False, modelcom=1, smooth=True, enableseg=False,
                 smoothseg=True, detectioncon=0.5, trackcon=0.5):
        self.mode = mode
        self.modelcom = modelcom
        self.smooth = smooth
        self.enableseg = enableseg
        self.smoothseg = smoothseg
        self.detectioncon = detectioncon
        self.trackcon = trackcon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.modelcom, self.smooth,
                                      self.enableseg, self.smoothseg,
                                      self.detectioncon, self.trackcon)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                            self.mpPose.POSE_CONNECTIONS)
        return img
    
    def findPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lmList

detector = poseDetector()
cap = cv2.VideoCapture(0)

def gen_frames():  # Generate frame by frame from camera
    while True:
        success, frame = cap.read()  # Read from video file
        if not success:
            break
        else:
            frame = detector.findPose(frame)
            lmList = detector.findPosition(frame)
            if lmList:
                cv2.circle(frame, (lmList[12][1], lmList[12][2]), 25, (0, 0, 255), cv2.FILLED)

            ret, buffer = cv2.imencode('.jpg', cv2.resize(frame, (0,0), fx=0.5, fy=0.5))
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

    
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
