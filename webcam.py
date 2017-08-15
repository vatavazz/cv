import numpy as np
import cv2
import time
from datetime import datetime
from facedetector import FaceDetector

# in seconds
VID_LENGTH = 10
MIN_AREA = 250
FOURCC = cv2.VideoWriter_fourcc(*'XVID')
fd = FaceDetector("haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)


concluded = False
# firstFrame = cv2.imread("background.jpg")
firstFrame = None
startTime = None

LOOPS = 5

while True:
	ret, frame = cap.read()
	grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(grey, (21, 21), 0)
	
	if firstFrame is None:
		firstFrame = blur
	
	# face detetcion
	# faces = frame
	# faceRects = fd.detect(grey, scaleFactor = 1.1, minNeighbors = 5, minSize = (30,30))
	# for (x,y,w,h) in faceRects:
		# cv2.rectangle(faces, (x,y), (x+w, y+h), (0, 255, 0), 2)
	# cv2.imshow("faces video", faces)
	
	# motion detection
	frameDelta = cv2.absdiff(firstFrame, blur)
	thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
	thresh = cv2.dilate(thresh, None, iterations=2)
	diff = cv2.subtract(firstFrame, blur)
	_, cnts, _= cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	
	for c in cnts:
		if cv2.contourArea(c) < MIN_AREA:
			continue
		
		(x,y,w,h) = cv2.boundingRect(c)
		cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
	
	cv2.imshow("motion video", frame)
	
	if (len(cnts) > 0) and (startTime == None):
		startTime = time.time()
		print("detected motion, saving to file")
		capname = "{}.avi".format(str(datetime.now().isoformat()))
		capname = capname.replace(":","-")
		vid = cv2.VideoWriter(capname, FOURCC, 5, (640, 480))
	
	if startTime != None:
		if (time.time() - startTime)< VID_LENGTH:
			vid.write(frame)
		elif concluded == False:
			vid.release()
			print("finished recording")
			concluded = True
			startTime = None
	
	
	k=cv2.waitKey(10)& 0xff
	# if k == 32:
		# capname = "{}.jpg".format(str(datetime.now().isoformat()))
		# capname = capname.replace(":","-")
		# cv2.imwrite(capname, blur)
	if k == 27:
		break

cap.release()
if vid != None:
	vid.release()
cv2.destroyAllWindows()