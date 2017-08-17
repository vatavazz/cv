import numpy as np
import cv2
import time
from datetime import datetime

# in seconds
VID_LENGTH = 10
# size of detected motion; experiment
MIN_AREA = 250
# codec
FOURCC = cv2.VideoWriter_fourcc(*'XVID')

cap = cv2.VideoCapture(0)

concluded = False
firstFrame = None
startTime = None

while True:
	ret, frame = cap.read()
	grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(grey, (21, 21), 0)
	
	if firstFrame is None:
		firstFrame = blur
	
	# motion detection
	frameDelta = cv2.absdiff(firstFrame, blur)
	thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
	thresh = cv2.dilate(thresh, None, iterations=2)
	diff = cv2.subtract(firstFrame, blur)
	# win fix, experiment
	_, cnts, _= cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	
	for c in cnts:
		if cv2.contourArea(c) >= MIN_AREA:
			(x,y,w,h) = cv2.boundingRect(c)
			cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
	
	cv2.imshow("Video Feed", frame)
	
	if (len(cnts) > 0) and (startTime == None):
		startTime = time.time()
		print("detected motion, recording video")
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
	if k == 27:
		break

if vid != None:
	vid.release()
cap.release()
cv2.destroyAllWindows()
