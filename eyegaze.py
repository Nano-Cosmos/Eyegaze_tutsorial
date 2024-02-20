import cv2
import numpy as np
import dlib

cap = cv2.VideoCapture(1)
#cap = cv2.VideoCapture('http://192.168.20.4:8080/video')
#cap = cv2.VideoCapture('http://192.168.43.1:8080/video')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def mid_point(p1,p2):
    return int((p1.x + p2.x)/2),int((p1.y + p2.y)/2)


while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        x,y = face.left(),face.top()
        x1,y1 = face.right(),face.bottom()
        cv2.rectangle(frame,(x,y), (x1,y1), (0,255,0), 2)

        landmarks = predictor(gray,face)
        #x = landmarks.part(36).x
        #y = landmarks.part(36).y
        #cv2.circle(frame,(x,y), 3,(0,0,255),2 )
        left_point = (landmarks.part(36).x, landmarks.part(36).y)
        right_point = (landmarks.part(39).x, landmarks.part(39).y)
        center_top = mid_point(landmarks.part(37),landmarks.part(38))
        center_bottom = mid_point(landmarks.part(41),landmarks.part(40))

        hor_line = cv2.line(frame, left_point, right_point, (0,255,0), 2)
        ver_line = cv2.line(frame,center_top,center_bottom,(0,255,0),2)



    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame,(640, 480))
    cv2.imshow("frame", frame)
    key = cv2.waitKey(1)
    if key ==  27:
        break

cap.release()
cv2.destroyAllWindows()
