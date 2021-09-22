from operator import delitem
import random
import os
import numpy as np
from numpy import save
import tensorflow as tf
import dlib
import cv2
from scipy.spatial.distance import cosine
from tensorflow.keras.models import load_model
from sklearn.preprocessing import Normalizer
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("shape.dat")
import logging
logger=logging.getLogger('my_logger')
logger.propagate=False
logger.disabled=True


def reset_random_seeds():
   os.environ['PYTHONHASHSEED']=str(1)
   tf.random.set_seed(1)
   np.random.seed(1)
   random.seed(1)
def normalize(img):
    mean,std=img.mean(),img.std()
    return (img-mean) / std


class Main():
    def __init__(self):
        self.T=np.arange(0,1,0.01)
        self.Thresholds=[]
        self.Logs=[]
        self.Fold=0
        for x in range(9):
            main_image_1=cv2.imread(f"images//{self.Fold}.jpg")
            main_image_1=cv2.cvtColor(main_image_1,cv2.COLOR_BGR2RGB)
            main_image_2=cv2.imread(f"images//{self.Fold+1}.jpg")
            main_image_2=cv2.cvtColor(main_image_2,cv2.COLOR_BGR2RGB)

            face_1=detector(main_image_1,1)[0]  
            face_2=detector(main_image_2,1)[0]
            (x1,y1)=face_1.left(),face_1.top()
            (x2,y2)=face_1.right(),face_1.bottom()
            img_roi_1=main_image_1[y1:y2,x1:x2]
            (x1,y1)=face_2.left(),face_2.top()
            (x2,y2)=face_2.right(),face_2.bottom()
            img_roi_2=main_image_2[y1:y2,x1:x2]

            img_1=cv2.resize(img_roi_1,(160,160))
            img_2=cv2.resize(img_roi_2,(160,160))
            img_1=img_1 / 255.
            img_2=img_2 / 255.
            img_1=img_1[np.newaxis,:]
            img_2=img_2[np.newaxis,:]
            face_model=load_model('facenet_keras.h5')
            encode_1=face_model.predict(img_1)[0]
            encode_2=face_model.predict(img_2)[0]

            N=Normalizer('l2')
            N_encode_1=N.transform(np.expand_dims(encode_1,axis=0))[0]
            N_encode_2=N.transform(np.expand_dims(encode_2,axis=0))[0]

            dist=cosine(N_encode_1,N_encode_2)
            self.Fold+=2
            self.Thresholds.append(dist)
            for x in self.T:
                if dist < x:
                    # 1 means OK
                    self.Logs.append('1')
                else: # 0 means Diffrent
                    self.Logs.append('0')
        A=np.array(self.Logs)
        B=np.split(A,9) # Tensors For Test The Threshold 
        for x in range(9):
            save(f"Logs//Tensors{x}",B[x])
        # You Can Print The B variable To See That
        Mean=np.mean(self.Thresholds)
        Std=np.std(self.Thresholds) 
        #print( "Mean :",Mean)
        #print("STD :",Std)


    
    def Verify(self,img_1,img_2):
        try:
            main_image_1=cv2.imread(img_1)
            main_image_1=cv2.cvtColor(main_image_1,cv2.COLOR_BGR2RGB)
            main_image_2=cv2.imread(img_2)
            main_image_1=cv2.cvtColor(main_image_1,cv2.COLOR_BGR2RGB)
            face_1=detector(main_image_1,1)[0]
            face_2=detector(main_image_2,1)[0]
            
            (x1,y1)=face_1.left(),face_1.top()
            (x2,y2)=face_1.right(),face_1.bottom()
            img_roi_1=main_image_1[y1:y2,x1:x2]

            (x1,y1)=face_2.left(),face_2.top()
            (x2,y2)=face_2.right(),face_2.bottom()
            img_roi_2=main_image_2[y1:y2,x1:x2]


            img_1=cv2.resize(img_roi_1,(160,160))
            img_2=cv2.resize(img_roi_2,(160,160))
            img_1=img_1 / 255.
            img_2=img_2 / 255.
            img_1=img_1[np.newaxis,:]
            img_2=img_2[np.newaxis,:]
            face_model=load_model('facenet_keras.h5')
            encode_1=face_model.predict(img_1)[0]
            encode_2=face_model.predict(img_2)[0]

            N=Normalizer('l2')
            N_encode_1=N.transform(np.expand_dims(encode_1,axis=0))[0]
            N_encode_2=N.transform(np.expand_dims(encode_2,axis=0))[0]

            dist=cosine(N_encode_1,N_encode_2)
            if dist > 0.5:
                print("Not Match !")
            else:
                print("Pics Are Match")
            print("Distance Is : {}".format(dist))
        except:
            pass






obj=Main()
obj.Verify('3.jpg','2.jpg')
