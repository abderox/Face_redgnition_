import cv2, numpy as np;
import xlwrite
import time
import sys
import os 
from playsound import playsound
start=time.time()
period=8
face_cas = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
# cap = cv2.VideoCapture(0,cv2.CAP_DSHOW);
recognizer = cv2.face.LBPHFaceRecognizer_create();
recognizer.read('trainner.yml');
flag = 0;
id=0;
filename='filename';
dict = {
            'item1': 1
}
#font = cv2.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 5, 1, 0, 1, 1)
font = cv2.FONT_HERSHEY_SIMPLEX

imagePaths=[os.path.join("attendance/folder",f) for f in os.listdir("attendance/folder")]
for imagePath in imagePaths:
        #loading the image and converting it to gray scale    
    print(imagePath)
    img = cv2.imread(imagePath).copy()
    cv2.imshow('frame',img);
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
    faces = face_cas.detectMultiScale(gray, 1.3, 7);
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0),2);
        id,conf=recognizer.predict(roi_gray)
        if(conf < 50):
         if(id==1):
            id='abdelhadi mouzafir'
            if((str(id)) not in dict):
                filename=xlwrite.output('attendance','class1',1,id,'yes');
                dict[str(id)]=str(id);
                
         elif(id==2):
            id = 'no lo see'
            if ((str(id)) not in dict):
                filename =xlwrite.output('attendance', 'class1', 2, id, 'yes');
                dict[str(id)] = str(id);

         elif(id==3):
            id = 'Mouzafir 100% '
            if ((str(id)) not in dict):
                filename =xlwrite.output('attendance', 'class1', 3, id, 'yes');
                dict[str(id)] = str(id);

         elif(id==5):
            id = 'me'
            if ((str(id)) not in dict):
                filename =xlwrite.output('attendance', 'class1', 4, id, 'yes');
                dict[str(id)] = str(id);

        else:
             id = 'Unknown, can not recognize'
             filename =xlwrite.output('attendance', 'class1', 4, id, 'yes');
             flag=flag+1
             break
        
        cv2.putText(img,str(id)+" "+str(conf),(x,y-10),font,0.55,(120,255,120),1)
        #cv2.cv.PutText(cv2.cv.fromarray(img),str(id),(x,y+h),font,(0,0,255));
    while(True):
        cv2.imshow('frame',img);
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break;
    #cv2.imshow('gray',gray);
    # if flag == 10:
    #     playsound('transactionSound.mp3')
    #     print("Transaction Blocked")
    #     break;
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break;

#cap.release();
cv2.destroyAllWindows();
