"coding utf-8"


import cv2 as cv 
#import numpy as np 
#import face_recognition as fr

from functions import erase_face2


""" 
Variables index :


"""



# First of all we gotta turn on the cam and make sure it works well


cap = cv.VideoCapture(0)
hasFrame, frame = cap.read()

print("\n\nPress q to quit.... \n\n")


while cv.waitKey(1) < 0:
   
    hasFrame, frame = cap.read()
    
    if not hasFrame:
        cv.waitKey()
        print("Camera failed to start")
        break


    frame = erase_face2(frame)

    cv.imshow("cc", frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):	#press q to quit
    	break