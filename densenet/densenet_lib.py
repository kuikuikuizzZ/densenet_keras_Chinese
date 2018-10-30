# coding: utf-8

import os 
import sys
sys.path.insert(0, os.getcwd())

from . import  keys
from . import densenet
import difflib
import numpy as np
from PIL import Image,ImageOps
from keras.layers import Input
from keras.models import Model
from imp import reload

characters = keys.alphabet_union[:]
characters = characters[1:] + u'卍'


# In[3]:

model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'models/model_crnn_gru_v1.h5')
# print(os.path.dirname(os.path.dirname(__file__)))
# print(model_path)
# print(os.path.abspath(__file__))



class Densenet_keras(object):
    def __init__(self,model_path=model_path,characters=characters,height=32):
        self.height = height
        self.nClass = len(characters)
        self.characters = characters
        self.basemodel = self.get_model(height=self.height,nClass=self.nClass)
        self.basemodel.load_weights(model_path)
        
    def get_model(self,height=32,nClass=len(characters)):
        input_tensor = Input(shape=(height,None,1),name='the_input')
        y_pred = densenet.dense_cnn(input_tensor, nClass)
        basemodel = Model(inputs=input_tensor,outputs=y_pred)
        return basemodel
        
    def recognize(self,img):
        im = Image.fromarray(img)
        im = im.convert('L')
        scale = im.size[1] * 1.0 / 32
        w = int(im.size[0] / scale)
        im = im.resize((w, 32),Image.ANTIALIAS)
        img = np.array(im).astype(np.float32) / 255.0 - 0.5
#         print(img.shape,scale)
        X = img.reshape((32, w, 1))
        X = np.array([X])
        print(X.shape)
        y_pred = self.basemodel.predict(X)
        y_pred = y_pred[:, :, :]
        out = self.decode(y_pred)  ##
        return out
    
    def decode(self,pred):
        text = pred.argmax(axis=2)[0]
        length = len(text)
        char_list = []
        n = self.nClass-1
        for i in range(length):
            # 这里text[i] != n就是处理出来的分隔符，text[i - 1] == text[i]处理重复的字符
            if text[i] != n and (not (i > 0 and text[i - 1] == text[i])):
                    char_list.append(self.characters[text[i]])
        return u''.join(char_list)
    
    def load_weights(self,model_path):
        self.basemodel.load_weights(model_path)
        print('loaded model')
    
    

if __name__ == '__main__':
    import cv2
    densenet_keras = Densenet_keras()
    img_path = './test/"依那普利氢氯噻嗪.png'
    image = cv2.imread(img_path)
    text = densenet_keras.recognize(image)
    print(text)




