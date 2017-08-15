import numpy as np
import cv2
from datetime import datetime as time
from facedetector import FaceDetector

fd = FaceDetector("haarcascade_frontalface_default.xml")


cap = cv2.VideoCapture(0)

while True:
	ret,img = cap.read()
	# cv2.imshow("original video", img)
	
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# cv2.imshow("gray video", gray)
	
	faces = img
	faceRects = fd.detect(gray, scaleFactor = 1.1, minNeighbors = 5, minSize = (30,30))
	for (x,y,w,h) in faceRects:
		cv2.rectangle(faces, (x,y), (x+w, y+h), (0, 255, 0), 2)
	cv2.imshow("faces video", faces)
	
	k=cv2.waitKey(10)& 0xff
	
	if k == 32:
		capname = "{}.jpg".format(str(time.now().isoformat()))
		# name doesnt accept :
		capname = capname.replace(":","-")
		cv2.imwrite(capname, img)
	elif k == 27:
		break
cap.release()
cv2.destroyAllWindows()