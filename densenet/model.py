#-*- coding:utf-8 -*-
import os
import sys
import numpy as np
from imp import reload
from PIL import Image, ImageOps

from keras.layers import Input
from keras.models import Model

sys.path.append(os.path.dirname(os.getcwd()))
import  densenet.keys as keys
import densenet.densenet as densenet
import densenet.keys_keras as keys_keras
reload(densenet)

characters = keys_keras.alphabet_union[:]
characters = characters[1:] + u'卍'



def densenet_cnn_model(height=32,nClass=len(characters)):
    input_tensor = Input(shape=(height,None,1),name='the_input')
    y_pred = densenet.dense_cnn(input_tensor, nClass)
    basemodel = Model(inputs=input_tensor,outputs=y_pred)
    return basemodel
        

def densenet_rnn_model(height=32,nClass=len(characters)):    
    input_tensor = Input(shape=(height, None, 1), name='the_input')
    y_pred = densenet.dense_rnn(input_tensor, nClass)
    basemodel = Model(inputs=input_tensor, outputs=y_pred)
    return basemodel