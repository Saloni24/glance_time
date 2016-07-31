import cv2
import numpy as np
import time



face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
final=0
tot=[]
total=0

cap = cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    face= face_cascade.detectMultiScale(gray,1.3,5)
    if len(face)==0:
	
	final=0
    if  len(face)!=0:

        for (x,y,w,h) in face:
	    t1=time.time()
	   
	#print t1
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),1)
            roi_image=gray[y:y+h, x:x+w]
	    roi_gray=frame[y:y+h, x:x+w]
	    smile=smile_cascade.detectMultiScale(roi_image,1.3,5)
            for tx,ty,tw,th in smile:
	        
    		t2=time.time()

    		total = t2-t1 
    		final=final+total
    
    #if  len(face)!=0:
	        print final
	
    
    cv2.putText(frame, 'Total glance time: %r' %final,(30,28), cv2.FONT_HERSHEY_TRIPLEX,1, (0,255,0), 2)
    cv2.putText(frame, 'TetherBox Technologies',(415,464), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,0,255), 2)
    cv2.imshow('smile',frame)
    k=cv2.waitKey(5)
    if k==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
     	
