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
import time


from keras.layers import Input
from keras.models import Model
from imp import reload
from keras.backend import ctc_decode,eval
from PIL import Image,ImageOps
from keras.utils import  multi_gpu_model
from tensorflow.python.platform import gfile

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
            self.config.gpu_options.per_process_gpu_memory_fraction = 0.4
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


class Densenet_tf(object):
    def __init__(self,pb_model_path,characters=characters,height=32,fixed_batch_size=512):
        self.height = height
        self.nClass = len(characters)
        self.characters = characters
        self.sess = self.load_pb(pb_model_path)
        # input_tensor
        self.input= self.sess.graph.get_tensor_by_name('the_input:0')
        # output_tensor
        self.output = self.sess.graph.get_tensor_by_name('out/truediv:0')
        self.fixed_batch_size=fixed_batch_size
#         self.input.set_shape([fixed_batch_size,32,800,1])
        init_data = np.zeros([self.fixed_batch_size,32,800,1])
        self.sess.run(self.output, {self.input: init_data})

    def load_pb(self,pb_file_path):
        tf.reset_default_graph()
        sess = tf.Session()
        with gfile.FastGFile(pb_file_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')
        return sess

                
    def recognize(self,img):
        img_L = img.astype(np.float32)
        img_L = img_L[:,:,0]*114/1000 + img_L[:,:,1]* 587/1000 + img_L[:,:,2]* 299/1000 
        scale = img_L.shape[0]*1.0/32.0
        w = int(img_L.shape[1]/scale)
        img = cv2.resize(img_L,(w,32),cv2.INTER_AREA)
        img = np.array(img)/ 255.0 - 0.5
        X = img.reshape((32, w, 1))
        X = np.array([X])
        y_pred = self.sess.run(self.output, {self.input: X})
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
#         start = time.time()
        y_pred = self.sess.run(self.output, {self.input: X})
        return y_pred
    
    def decode(self,pred,with_confidence=False):
        text = pred.argmax(axis=2)[0]
#         print(text)
        length = text.shape[0]
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
#         start = time.time()
        y_pred = self.sess.run(self.output, {self.input: X})
        y_pred = y_pred[:, :, :]
        out = self.decode(y_pred,with_confidence=True)  ##
        return out
        
        
    def recognize_on_batch(self,img_list,with_confidence=False,fixed_size=False):
        # fixed dimension size version preprocess function
        assert len(img_list) <= self.fixed_batch_size
        if fixed_size:
            img_batch_fixed = np.ones([self.fixed_batch_size,32,800,1])*0.5
            for i,img in enumerate(img_list):
                img_L = img.astype(np.float32)
                if len(img_L.shape) ==3:
                    img_L = img_L[:,:,0]*114/1000 + img_L[:,:,1]* 587/1000 + img_L[:,:,2]* 299/1000 
                scale = img_L.shape[0]*1.0/32.0
                w = int(img_L.shape[1]/scale)
                w = 800 if w >= 800 else w
                img = cv2.resize(img_L,(w,32),cv2.INTER_AREA)
                img = img/ 255.0 - 0.5
                img = img.reshape((32, w, 1))
                img_batch_fixed[i,:,:w,:] = img          
        else :
            ### flexible version preprocess function
            max_w = int(max([32.0/img.shape[0]*img.shape[1] for img in img_list]))
            img_batch_flexible = np.ones([len(img_list),32,max_w,1])*0.5
#             print('img_batch ', img_batch_flexible.shape)
            for i,img in enumerate(img_list):
                img_L = img.astype(np.float32)
                if len(img_L.shape) ==3:
                    img_L = img_L[:,:,0]*114/1000 + img_L[:,:,1]* 587/1000 + img_L[:,:,2]* 299/1000 
                scale = img_L.shape[0]*1.0/32.0
                w = int(img_L.shape[1]/scale)
                img = cv2.resize(img_L,(w,32),cv2.INTER_AREA)
                img = img/ 255.0 - 0.5
                img = img.reshape((32, w, 1))
                img_batch_flexible[i,:,:w,:] = img
        img_batch = img_batch_fixed if fixed_size else img_batch_flexible
            
        assert len(img_batch.shape) == 4
        start = time.time()
        y_pred = self.sess.run(self.output, {self.input: img_batch})
        end = time.time()
        y_pred = y_pred[:len(img_list), :, :]
        out_list = []
        for i in range(y_pred.shape[0]):
            y_encode = np.array([y_pred[i,:,:]]) 
            out = self.decode(y_encode,with_confidence)  ##
            out_list.append(out)
        return out_list
        

class Densenet_tf_multi_gpus(object):
    def __init__(self,pb_model_path,characters=characters,height=32,width=800,fixed_batch_size=512):
        self.height = height
        self.width = width
        self.nClass = len(characters)
        self.characters = characters
        self.graph = tf.Graph()
        self.fixed_batch_size=fixed_batch_size
        # output_tensor
        self.sess = self.load_multi_gpu_model(pb_model_path)
        init_data = np.zeros([self.fixed_batch_size,self.height,self.width,1])
        self.sess.run([self.output_value,self.output_index], {self.input: init_data})

    def load_multi_gpu_model(self,pb_file_path):
        config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
        sess = tf.Session(config=config)
        devices = [x.name for x in sess.list_devices()][1:]
        self.devices = ['/' + ':'.join(name.lower().replace('/', '').split(':')[-2:]) for name in devices]
        self.gpus = [name for name in self.devices if 'gpu' in name]
        self.num_gpus = len(self.gpus)
        self.input = tf.placeholder(tf.float32, [None, self.height , None,1], name="input")
        input_splits = tf.split(self.input,self.num_gpus)
        result_value = []
        result_index = []
        counter = 0
        for d in range(self.num_gpus):
            with tf.device('/gpu:%s' % d):
                with tf.name_scope('%s_%s' % ('tower', d)):
                    with gfile.FastGFile(pb_file_path, 'rb') as f:
                        graph_def = tf.GraphDef()
                        graph_def.ParseFromString(f.read())
                        # input_tensor
                        output = tf.import_graph_def(graph_def, name='',
                                                          input_map={'the_input:0':input_splits[counter]},
                                                          return_elements=['out/truediv:0'])
                        counter += 1
                        with tf.variable_scope('result'):
                            output_value = tf.reduce_max(output[0],2)
                            output_index = tf.argmax(output[0],2)
                            result_value.append(output_value)
                            result_index.append(output_index)
                            tf.get_variable_scope().reuse_variables()
        self.output_value = tf.concat(result_value,axis=0)
        self.output_index = tf.concat(result_index,axis=0)
#         self.output = tf.stack(result,axis=0)
                                                       
        return sess
                
        
    def recognize_on_batch(self,img_list,with_confidence=False,fixed_size=False):
        # fixed dimension size version preprocess function
        assert len(img_list) <= self.fixed_batch_size
        if fixed_size:
            img_batch_fixed = np.ones([self.fixed_batch_size,32,self.width,1])*0.5
            for i,img in enumerate(img_list):
                img_L = img.astype(np.float32)
                if len(img_L.shape) ==3:
                    img_L = img_L[:,:,0]*114/1000 + img_L[:,:,1]* 587/1000 + img_L[:,:,2]* 299/1000 
                scale = img_L.shape[0]*1.0/32.0
                w = int(img_L.shape[1]/scale)
                w = self.width if w >= self.width else w
                img = cv2.resize(img_L,(w,32),cv2.INTER_AREA)
                img = img/ 255.0 - 0.5
                img = img.reshape((32, w, 1))
                img_batch_fixed[i,:,:w,:] = img          
        else :
            ### flexible version preprocess function
            max_w = int(max([32.0/img.shape[0]*img.shape[1] for img in img_list]))
            input_size = (len(img_list)+self.num_gpus-1)//self.num_gpus*self.num_gpus
            img_batch_flexible = np.ones([input_size,32,max_w,1])*0.5
            for i,img in enumerate(img_list):
                img_L = img.astype(np.float32)
                if len(img_L.shape) ==3:
                    img_L = img_L[:,:,0]*114/1000 + img_L[:,:,1]* 587/1000 + img_L[:,:,2]* 299/1000 
                scale = img_L.shape[0]*1.0/32.0
                w = int(img_L.shape[1]/scale)
                img = cv2.resize(img_L,(w,32),cv2.INTER_AREA)
                img = img/ 255.0 - 0.5
                img = img.reshape((32, w, 1))
                img_batch_flexible[i,:,:w,:] = img
        img_batch = img_batch_fixed if fixed_size else img_batch_flexible
            
        assert len(img_batch.shape) == 4
        start = time.time()
        y_pred,confidence = self.sess.run([self.output_index,self.output_value], {self.input: img_batch})
        end = time.time()
        print(end-start)
        y_pred = y_pred[:len(img_list), :]
        confidence = confidence[:len(img_list),:]
        out_list = []
        for i in range(y_pred.shape[0]):
            y_encode = y_pred[i,:]
            y_confidence = confidence[i,:]
            decoded = self.decode(y_encode,y_confidence,with_confidence)  ##
            out_list.append(decoded)
        return out_list
        
    def decode(self,text,pred_max,with_confidence=False):
        length = text.shape[0]
        char_list = []
        index_list = []
        n = self.nClass-1
        for i in range(length):
            # 这里text[i] != n就是处理出来的分隔符，text[i - 1] == text[i]处理重复的字符
            if text[i] != n and (not (i > 0 and text[i - 1] == text[i])):
                    char_list.append(self.characters[text[i]])
                    index_list.append(i)
        if with_confidence:            
            confidence = self.process_confidence(pred_max,index_list)
            return u''.join(char_list),confidence
        return u''.join(char_list)
        
    def process_confidence(self,pred_max,index_list):        
        confidence_list = [] 
        length = pred_max.shape[0]
        confidence_list = pred_max[index_list]
        length_confidence = len(confidence_list)

        if length_confidence < 3:
            confidence = np.mean(confidence_list)
        else :
            sorted_confidence = np.sort(confidence_list)
            confidence = np.mean(sorted_confidence[:length_confidence//2])
        return confidence
        
        


if __name__ == '__main__':
    import cv2
    densenet_keras = Densenet_keras()
    img_path = './test/"依那普利氢氯噻嗪.png'
    image = cv2.imread(img_path)
    text = densenet_keras.recognize(image)
    print(text)







