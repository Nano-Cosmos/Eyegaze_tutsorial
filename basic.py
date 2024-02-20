import cv2
import sys
import numpy as np

#Reading and displaying an image
"""img = cv2.imread("heist_fam.jpg",1)
cv2.imshow("Window",img)
if img is None:
    sys.exit("Couldn't load the image ")

k = cv2.waitKey(0)

if k == ord('s'):
    cv2.imwrite("heist_fam2.jpg",img)"""

#video capturing and playing
"""cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("couldn't open the camera")
    exit()
while True:
    ret, frame = cap.read()

    if not ret:
        print("Error in reading frames ")
        break;
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.imshow("Window",gray)
    k = cv2.waitKey(1)

    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()"""

#video saving

"""cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc,20.0,(640, 480))

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("error in reading frames ")
        break
    frame = cv2.flip(frame,1)
    out.write(frame)

    cv2.imshow("Video",frame)
    k = cv2.waitKey(1)

    if k == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()"""

#Drawing Geometrical shapes and adding text into image
# Create a black image
"""img = np.zeros((512,512,3), np.uint8)
cv2.line(img,(0,0),(511,511),(255,0,0),5)
cv2.rectangle(img,(384,0),(510,128),(0,255,0),3)
cv2.circle(img,(447,63), 63, (0,0,255), -1)
cv2.ellipse(img,(256,256),(100,50),0,0,180,255,-1)
pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
pts = pts.reshape((-1,1,2))
cv2.polylines(img,[pts],True,(0,255,255))
font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
cv2.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2,cv2.LINE_AA)

cv2.imshow("frame",img)
k = cv2.waitKey(0)

if k == ord('q'):
    cv2.imwrite('drawing.jpg',img)"""



