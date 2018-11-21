# coding: utf-8

import os 
import sys
sys.path.append(os.path.dirname(os.getcwd()))

import  densenet.keys as keys
import densenet.densenet as densenet
import difflib
import numpy as np
import cv2
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

from keras.layers import Input
from keras.models import Model
from imp import reload
from keras.backend import ctc_decode,eval
from PIL import Image,ImageOps
from keras.utils import  multi_gpu_model
import time
characters = keys.alphabet_union[:]
characters = characters[1:] + u'卍'


# In[3]:

model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'models/model_crnn_gru_v1.h5')
# print(os.path.dirname(os.path.dirname(__file__)))
# print(model_path)
# print(os.path.abspath(__file__))



class Densenet_keras(object):
    def __init__(self,get_model_function,model_path=model_path,characters=characters,height=32):
        self.height = height
        self.nClass = len(characters)
        self.characters = characters
        self.gpus = KTF._get_available_gpus()
        if self.gpus:
            self.config = tf.ConfigProto()
            self.config.gpu_options.per_process_gpu_memory_fraction = 0.95
            self.session = tf.Session(config=self.config)
            KTF.set_session(self.session)
            if len(self.gpus)==1:
                self.basemodel = get_model_function(height=self.height,nClass=self.nClass)
                self.basemodel.load_weights(model_path)
                self.predict_batch=512
            else:
                with tf.device('/cpu:0'):
                    model_template = get_model_function(height=self.height,nClass=self.nClass)
                    model_template.load_weights(model_path)
                self.basemodel = multi_gpu_model(model_template,len(self.gpus))
                self.predict_batch=len(self.gpus)*512
        else:
            self.basemodel = get_model_function(height=self.height,nClass=self.nClass)
            self.basemodel.load_weights(model_path)
        

        
    def recognize(self,img):
        img_L = img.astype(np.float32)
        img_L = img_L[:,:,0]*114/1000 + img_L[:,:,1]* 587/1000 + img_L[:,:,2]* 299/1000 
        scale = img_L.shape[0]*1.0/32.0
        w = int(img_L.shape[1]/scale)
        img = cv2.resize(img_L,(w,32),cv2.INTER_AREA)
        img = np.array(img)/ 255.0 - 0.5
        X = img.reshape((32, w, 1))
        X = np.array([X])
        KTF.set_session(self.session)
        y_pred = self.basemodel.predict(X)
        y_pred = y_pred[:, :, :]
        out = self.decode(y_pred)  ##
        return out
    
    
    def predict(self,img):        
        im = Image.fromarray(img)
        im = im.convert('L')
        scale = im.size[1] * 1.0 / 32
        w = int(im.size[0] / scale)
        im = im.resize((w, 32),Image.ANTIALIAS)
        img = np.array(im).astype(np.float32) / 255.0 - 0.5
        X = img.reshape((32, w, 1))
        X = np.array([X])
        start = time.time()
        KTF.set_session(self.session)
        y_pred = self.basemodel.predict(X)
        return y_pred
    
    def decode(self,pred,with_confidence=False):
        text = pred.argmax(axis=2)[0]
        length = len(text)
        char_list = []
        n = self.nClass-1
        for i in range(length):
            # 这里text[i] != n就是处理出来的分隔符，text[i - 1] == text[i]处理重复的字符
            if text[i] != n and (not (i > 0 and text[i - 1] == text[i])):
                    char_list.append(self.characters[text[i]])
                 
        if with_confidence:
            pred_max = pred.max(axis=2)[0]
            confidence_list = [] 
            for i in range(length):
                # 这里text[i] != n就是处理出来的分隔符，text[i - 1] == text[i]处理重复的字符
                if text[i] != n and (not (i > 0 and text[i - 1] == text[i])):    
                        confidence_list.append(pred_max[i])
            length_confidence = len(confidence_list)
            
            if length_confidence < 3:
                confidence = np.mean(confidence_list)
            else :
                sorted_confidence = np.sort(confidence_list)
                confidence = np.mean(sorted_confidence[:length_confidence//2])
            return u''.join(char_list),confidence
        return u''.join(char_list) 
    
    def recognize_with_confidence(self,img):
        im = Image.fromarray(img)
        im = im.convert('L')
        scale = im.size[1] * 1.0 / 32
        w = int(im.size[0] / scale)
        im = im.resize((w, 32),Image.ANTIALIAS)
        img = np.array(im).astype(np.float32) / 255.0 - 0.5
        X = img.reshape((32, w, 1))
        X = np.array([X])
        start = time.time()
        KTF.set_session(self.session)
        y_pred = self.basemodel.predict(X)
        y_pred = y_pred[:, :, :]
        out = self.decode(y_pred,with_confidence=True)  ##
        return out
        
    def load_weights(self,model_path):
        self.basemodel.load_weights(model_path)
        print('loaded model')
        
    def recognize_on_batch(self,img_list,with_confidence=False):
#         start = time.time()
        max_w = int(max([32.0/img.shape[0]*img.shape[1] for img in img_list]))
#         print(max_w)
#         assert len(img_list) < 128
        img_batch = np.ones([len(img_list),32,max_w,1])*0.5
        for i,img in enumerate(img_list):
            img_L = img.astype(np.float32)
            if len(img_L.shape) ==3:
                img_L = img_L[:,:,0]*114/1000 + img_L[:,:,1]* 587/1000 + img_L[:,:,2]* 299/1000 
            scale = img_L.shape[0]*1.0/32.0
            w = int(img_L.shape[1]/scale)
            img = cv2.resize(img_L,(w,32),cv2.INTER_AREA)
            img = img/ 255.0 - 0.5
            img = img.reshape((32, w, 1))
            img_batch[i,:,:w,:] = img
        end = time.time()
#         print('process use ' ,end-start)
        KTF.set_session(self.session)
#         print(time.time()-end)
        start = time.time()
        y_pred = self.basemodel.predict(img_batch,batch_size=self.predict_batch)
        end = time.time()
        print('process use ' ,end-start)
        y_pred = y_pred[:, :, :]
        out_list = []
        for i in range(y_pred.shape[0]):
            y_encode = np.array([y_pred[i,:,:]]) 
            out = self.decode(y_encode,with_confidence)  ##
            out_list.append(out)
        return out_list
            


if __name__ == '__main__':
    import cv2
    densenet_keras = Densenet_keras()
    img_path = './test/"依那普利氢氯噻嗪.png'
    image = cv2.imread(img_path)
    text = densenet_keras.recognize(image)
    print(text)







