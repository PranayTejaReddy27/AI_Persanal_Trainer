import cv2
import pandas as pd
import numpy as np
import time
import PoseModule as pm
from flask import Flask, render_template, Response

app = Flask(__name__)

def dumbbells():
    cap = cv2.VideoCapture(2)
    detector = pm.poseDetector()
    count = 0
    dir  = 0
    pTime = 0
    df = pd.DataFrame(columns=['position', '11x', '11y', 'angle', 'percentage','polarity'])

    while True:
        success, img = cap.read()
        #img = cv2.resize(img, (420,840))
        img = detector.findPose(img, False)
        lmList = detector.findPosition(img, False)
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        #print(lmList[])
        if len(lmList) != 0: 
            # left arm 
            angle = detector.findAngle(img, 11, 13, 15)
            # right arm
            #detector.findAngle(img, 12, 14, 16)
            per = np.interp(angle,(250,300),(0,100))
            bar = np.interp(angle,(250,300),(450,100))
            #print(angle,per)
            if per == 100 :
                if dir == 0:
                    count += 0.5
                    dir = 1

            if per == 0 :
                if dir == 1:
                    count += 0.5
                    dir = 0

            # Append new row of data to the DataFrame
            #new_data = {'position': "11,13,15", '11x': lmList[11][1], '11y': lmList[11][2], 'angle': angle, 'percentage': per, 'polarity': 1}
            #df = df._append(new_data, ignore_index=True)
                    

            #print(count)
            # accuracy bar 
            cv2.rectangle(img, (540,100),(580,450),(123,232,121),2)
            cv2.rectangle(img, (540,int(bar)),(580,450),(123,232,121),cv2.FILLED)
            cv2.putText(img, f'{int(per)} %', (540,75), cv2.FONT_HERSHEY_SIMPLEX,0.5, (23,34,10),2)
            # draw rectangle 
            cv2.rectangle(img, (20,10),(200,100),(123,232,121),cv2.FILLED)
            # count text display
            cv2.putText(img, str(int(count)), (90,90), cv2.FONT_HERSHEY_SIMPLEX,2, (23,34,10),3)

        # display FPS
        cv2.putText(img,"FPS : " + str(int(fps)), (70,30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
        ret, buffer = cv2.imencode('.jpg', cv2.resize(img, (0,0), fx=1, fy=1))
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

        key = cv2.waitKey(1)
        if key == ord('q'):
            break  # Exit the loop if 'q' is pressed
    #print(df)
    #df.to_csv('Dumbbells_DataSet4.csv', index=False, encoding='utf-8' )
    #img.release()  # Release the camera
    #cv2.destroyAllWindows()
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(dumbbells(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)

    
