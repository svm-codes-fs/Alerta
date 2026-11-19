import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time


mixer.init()
sound = mixer.Sound('alarm.wav')
sound.set_volume(0.3)   # 0.0 (mute) .. 1.0 (full)

face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')



lbl=['Close','Open']

try:
    model = load_model('models/cnncat2.h5')
except TypeError:
    # cnnCat2.h5 was saved by Keras 2.1.6 and its config does not load in Keras 3:
    # rebuild the same network as model.py and copy the stored weights in
    import h5py
    from keras.models import Sequential
    from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
    model = Sequential([Input((24,24,1)), Conv2D(32,(3,3),activation='relu'), MaxPooling2D((1,1)),
                        Conv2D(32,(3,3),activation='relu'), MaxPooling2D((1,1)), Conv2D(64,(3,3),activation='relu'),
                        MaxPooling2D((1,1)), Dropout(0.25), Flatten(), Dense(128,activation='relu'), Dropout(0.5),
                        Dense(2,activation='softmax')])
    w = h5py.File('models/cnnCat2.h5','r')['model_weights']
    model.set_weights([w[n][n][k][()] for n in ['conv2d_1','conv2d_2','conv2d_3','dense_1','dense_2']
                       for k in ('kernel:0','bias:0')])
path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count=0
score=0
thicc=2
rpred=[99]
lpred=[99]

while(True):
    ret, frame = cap.read()
    if not ret:
        print('Webcam stopped sending frames (is another app using it?)')
        break
    height,width = frame.shape[:2] 

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    left_eye = leye.detectMultiScale(gray)
    right_eye =  reye.detectMultiScale(gray)

    cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )

    for (x,y,w,h) in right_eye:
        r_eye=frame[y:y+h,x:x+w]
        count=count+1
        r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye,(24,24))
        r_eye= r_eye/255
        r_eye=  r_eye.reshape(24,24,-1)
        r_eye = np.expand_dims(r_eye,axis=0)
        #rpred = model.predict_classes(r_eye)
        # rpred = model.predict(r_eye)
        # rpred = np.round(rpred).astype(int)
        rpred = np.argmax(model(r_eye, training=False), axis=-1)
        if(rpred[0]==1):
            lbl='Open' 
        if(rpred[0]==0):
            lbl='Closed'
        break

    for (x,y,w,h) in left_eye:
        l_eye=frame[y:y+h,x:x+w]
        count=count+1
        l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
        l_eye = cv2.resize(l_eye,(24,24))
        l_eye= l_eye/255
        l_eye=l_eye.reshape(24,24,-1)
        l_eye = np.expand_dims(l_eye,axis=0)
        #lpred = model.predict_classes(l_eye)
        # lpred = model.predict(l_eye)
        # lpred = np.round(lpred).astype(int)
        lpred = np.argmax(model(l_eye, training=False), axis=-1)
        print(lpred)
        if(lpred[0]==1):
            lbl='Open'   
        if(lpred[0]==0):
            lbl='Closed'
        break

    if(rpred[0]==0 and lpred[0]==0):
        score=score+1
        cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    # if(rpred[0]==1 or lpred[0]==1):
    else:
        score=score-1
        cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    
        
    if(score<0):
        score=0   
    cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    if(score>15):
        #person is feeling sleepy so we beep the alarm
        cv2.imwrite(os.path.join(path,'image.jpg'),frame)
        try:
            sound.play()
            
        except:  # isplaying = False
            pass
        if(thicc<16):
            thicc= thicc+2
        else:
            thicc=thicc-2
            if(thicc<2):
                thicc=2
        cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc) 
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
