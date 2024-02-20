import numpy as np
import cv2

"""img = cv2.imread('lena.jpg', -1)

print(img)

cv2.imshow('image', img)
k = cv2.waitKey(0)

if k == 27:
    cv2.destroyAllWindows()
elif k == ord('s'):
    cv2.imwrite('lena_copy.jpg', img)
    cv2.destroyAllWindows()

# video capture and saving
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
while cap.isOpened():
    ret, frame = cap.read()
    if (ret == True):
        print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out.write(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()

#img =cv2.imread('heist_fam.jpg',1)
img = np.zeros([512, 512, 3], np.uint8)

img = cv2.line(img, (0, 0), (255, 255), (255, 0, 0), 10)
img = cv2.rectangle(img, (384, 0), (510, 128), (0, 0, 255), -1)
img = cv2.circle(img, (447,80), 63, (0, 255, 0), -1)
font = cv2.FONT_HERSHEY_SIMPLEX
img = cv2.putText(img, 'OpenCV', (100, 300), font, 4, (0, 255, 255), 10, cv2.LINE_AA)
cv2.imshow('frame', img)

cv2.waitKey(0)
cv2.destroyAllWindows()


# video capturing
cap = cv2.VideoCapture('http://192.168.20.3:8080/video')
while cap.isOpened():
    ret, frame = cap.read()
    if (ret == True):
        resized = cv2.resize(frame, (600, 400))
        cv2.imshow('frame', resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
              break
    else:
        break
cap.release()
cv2.destroyAllWindows()
"""

cap = cv2.VideoCapture(0)

tracker = cv2.legacy.TrackerMOSSE_create()
tracker = cv2.TrackerCSRT_create()
ret, frame = cap.read()
bbox = cv2.selectROI("Tracking",frame,False)
tracker.init(frame,bbox)

def drawbox(frame,bbox):
    x,y,w,h = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
    cv2.rectangle(frame,(x,y),((x+w),(y+h)),(255,0,255),3,1)
    cv2.putText(frame, "Tracking", (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


while True:
    timer = cv2.getTickCount()
    ret, frame = cap.read()

    ret,bbox = tracker.update(frame)
    print(bbox)

    if ret:
        drawbox(frame,bbox)
    else:
        cv2.putText(frame,"Lost", (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    fps = cv2.getTickFrequency()/(cv2.getTickCount()-timer)
    cv2.putText(frame,str(int(fps)),(75,50),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
    cv2.imshow("Tracking",frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break;
























