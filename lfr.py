import cv2
import numpy as np
import time

def roi(img,vertices):
	mask = np.zeros_like(img)
	cv2.fillPoly(mask,vertices,255)
	masked = cv2.bitwise_and(img,mask)
	return masked

def main():
	windowName='Live Video'
	filename = 'C:\\Users\\spars\\Desktop\\programs\\python\\opencv\\output\\lfr.avi'
	cap=cv2.VideoCapture(filename)
	cv2.namedWindow(windowName)
	#cap=cv2.VideoCapture(1)
	ret, frame = cap.read()   
	last_time = time.time()														#time



	while ret:
		ret, frame = cap.read()
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		#low = np.array([0,0,0])													#tracking using color
		#high = np.array([100,100,100])
		#image_mask = cv2.inRange(frame, 0, 95)
		#processed_img1 = cv2.bitwise_and(frame, frame, mask = image_mask)

		ret, processed_img3 = cv2.threshold(frame, 110, 255, cv2.THRESH_BINARY)		#thresholding

		Gaussian = cv2.GaussianBlur(processed_img3,(27,27),0)

		processed_img2 = cv2.Canny(Gaussian , threshold1 = 180 , threshold2 = 200)		#canny edge

		vertices = np.array([[0,300],[800,300],[800,200],[0,200]])
		r = roi(processed_img2,[vertices])


		print('loop took {} seconds'.format(time.time()-last_time))					#printing time
		last_time = time.time()
		cv2.imshow(windowName, frame)
		cv2.imshow('canny', processed_img2)
		cv2.imshow('track', processed_img3)
		cv2.imshow('Gaussian', Gaussian)
		cv2.imshow('roi', r)


		if cv2.waitKey(25) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			break

        
	cv2.destroyWindow(windowName)

	cap.release()

if __name__=="__main__":
	main()    

# pikachu