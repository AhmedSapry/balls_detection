import cv2 
import numpy as np
import cvzone 
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

################################################
frameWidth = 640 
frameHeight = 480 
cap =cv2.VideoCapture('balls_mia.mp4')
cap.set(3,frameWidth)
cap.set(4,frameHeight)
cap.set(10,150)
myClassifier = cvzone.Classifier('keras_model_mia.h5','labels_mia.txt')

myColors = [[84,95,144,159,255,255],
            [168,121,82,181,224,255]
           ]
colorValues = [[255,0,0],   #bgr
               [0,0,255]]

StringColor = ["blue_ball","red_ball"]

################################################

def findColor(img,myColors,colorValues):
    max_index=10
    imgHsv =cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    count = 0 
    font = cv2.FONT_HERSHEY_SIMPLEX
    for color in myColors :
        lower = np.array(color[0:3])
        upper = np.array(color[3:])
        mask = cv2.inRange(imgHsv,lower,upper)
        x,y,w,h = getContours(mask)
        cropped_img =img[y:y+h+10, x:x+w+10]
        if(cropped_img.shape[0]>0):
            cv2.imshow('cropped_img',cropped_img)
            cropped_img = cv2.resize(cropped_img,(224,224))
            max_index = detection(cropped_img)
        #Circule_detection(mask)
       
        print(max_index)
        if(max_index==1):
            #cv2.circle(imgResult,(x+w//2,y+h//2),int(w/2),colorValues[count],cv2.FILLED)
            cv2.rectangle(imgResult, (x,y), (x+w,y+h), colorValues[count], 5)
            cv2.putText(imgResult,StringColor[count],(x-5,y+5),font,0.5,(255,255,255),2,cv2.LINE_AA)
        count +=1 


def getContours(img):
    countours ,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    x, y , w,h = 0,0,0,0
    for cnt in countours : 
        area = cv2.contourArea(cnt)
        if area > 75 :
            #cv2.drawContours(imgResult,cnt,-1,(255,0,0),10)
            peri = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,0.01*peri,True)
            x , y , w, h = cv2.boundingRect(approx)
    return x,y,w,h

def detection(image_prediciton):
    
    #img = cv2.resize(image_prediciton,(512,512))
    predictions = myClassifier.getPrediction(image_prediciton)
    print(predictions[0])
    maximum = np.argmax(predictions[0])
    
    return maximum

    

while (cap.isOpened()) :
    ret, frame =cap.read()
    frame = cv2.resize(frame,(512,512))
    if ret== True :
        imgResult = frame.copy()
        findColor(frame,myColors,colorValues)
        cv2.imshow('imgResult',imgResult)
        if cv2.waitKey(1) & 0xFF ==ord('q') :
         break
    else :
        break
        
cap.release()
#out1.release()
cv2.destroyAllWindows()
