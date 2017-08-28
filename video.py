import numpy as np
import cv2
import os
	
def main():
	FOURCC = cv2.VideoWriter_fourcc(*'XVID')
	name = ""
	firstFrame = None
	text = "undetected"
	
	for filename in os.listdir("cuti"):
		print(filename)
		cap = cv2.VideoCapture("cuti/"+filename)
		start_avg = None
		vid = cv2.VideoWriter(filename+".avi", FOURCC, 25, (200, 200))
		while (cap.isOpened()):
			ret,img = cap.read()
			if ret:
				# resize for viewing
				show = cv2.resize(img, (640, 480))
				
				# convrt to hsv space
				# threshold colours
				# gray = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
				gray = cv2.threshold(img, 125, 255, cv2.THRESH_BINARY)[1]
				
				# add text and show video
				cv2.putText(show, "Status: {}".format(text), (10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
				cv2.imshow("original video", show)
				
				# average pixels in row, average rows
				avg_row = np.average(gray, axis=0)
				avg_col = np.average(avg_row, axis=0)
				avg_col8 = np.uint8(avg_col)
				avg_img = np.array([[avg_col8]*200]*200, np.uint8)
				cv2.imshow("a video", avg_img)
				vid.write(avg_img)
				
				if start_avg is None:
					start_avg = avg_col
				
				# print difference between current avg and first frame avg
				# potentially average colour over several frames
				# print("abs: {}".format(abs(start_avg-avg_col)))
				
				# text on image
				if abs(start_avg[0]-avg_col[0]) > 5 or abs(start_avg[1]-avg_col[1]) > 5 or abs(start_avg[2]-avg_col[2]) > 5:
					text = "detected"
				else:
					text = " "
				
				k=cv2.waitKey(10)& 0xff
				if k == 27:
					exit(0)
			else:
				break
		cap.release()
		cv2.destroyAllWindows()

if __name__=="__main__":
    main()
