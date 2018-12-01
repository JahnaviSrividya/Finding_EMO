

import os
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt

emotions=[0,1,5,6,7]
fishface=cv2.face.createFisherFaceRecognizer()
def getfiles(i):
    src="/media/hana/Windows/dataset/"+str(i)
    dir=os.listdir(src)
    random.shuffle(dir)
    train_data=dir[:int(len(dir)*0.8)]
    pred_data=dir[:int(len(dir)*0.2)]
    return train_data,pred_data

def make_sets():
    train_data=[]
    train_label=[]
    pred_data=[]
    pred_label=[]
    for i in emotions:
        training,prediction=getfiles(i)
        for file in training:
            image=cv2.imread("/media/hana/Windows/dataset/"+str(i)+"/"+file)
            gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            train_data.append(gray)
            train_label.append(i)
        for file in prediction:
            image=cv2.imread("/media/hana/Windows/dataset/"+str(i)+"/"+file)
            gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            pred_data.append(gray)
            pred_label.append(i)
    return train_data,train_label,pred_data,pred_label

def run_model():
    train_d,train_l,pred_d,pred_l=make_sets()
    print(pred_d)
    print("FisherFace Training Started")
    print("Training Size:"+str(len(train_d)))
    fishface.train(train_d,np.asarray(train_l))
    cnt=0
    correct=0
    incorrect=0
    for image in pred_d:
        image=np.asarray(image,dtype='float32')
        pred=fishface.predict(image)
        if pred==pred_l[cnt]:
            correct+=1
        else:
            incorrect+=1
        cnt+=1
    return ((100*correct)/(correct+incorrect))

result=[]
for i in range(10):
    correct=run_model()
    print(str(i)+": Accuracy:"+str(correct))
    result.append(correct)
print("Final:"+str(np.mean(result)))

