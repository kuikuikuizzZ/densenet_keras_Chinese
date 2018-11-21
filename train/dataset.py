
import os
import json
import numpy as np
from keras.utils import Sequence
from PIL import Image

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, dataset_dir,list_IDs, labels, batch_size=32, dim=(32,32,32), n_channels=1,
                 characters='', shuffle=True,maxLabelLength=10,mode='train'):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.characters = characters
        self.n_classes = len(self.characters)
        self.shuffle = shuffle
        self.on_epoch_end()
        self.dataset_dir = dataset_dir
        self.maxLabelLength = maxLabelLength
        self.mode = mode

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.zeros((self.batch_size, *self.dim, self.n_channels), dtype=np.float)
        Y = np.ones([self.batch_size, self.maxLabelLength],dtype=int) * 10000
        input_length = np.zeros([self.batch_size, 1])
        label_length = np.zeros([self.batch_size, 1])
        
        
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            try:
                img = Image.open(os.path.join(self.dataset_dir, ID)).convert('L')
                img = img.resize((self.dim[1],self.dim[0]))
                img = np.array(img, 'f') / 255.0 - 0.5
            except (OSError,IOError) as error:
                print(error)
                img = np.zeros(*self.dim,dtype=np.float)
                

            X[i,] = np.expand_dims(img, axis=2)
            
            label_origin = self.labels[ID]
            label_origin.replace(' ','')
            label = self.__one_hot(label_origin,length=len(label_origin))


            if(len(label) <= 0):
                print("%s label len < 0" %ID)
            # the input length for ctc_loss, for densenet pool size is about 8
            label_length[i] = len(label)
            input_length[i] = self.dim[1] // 8
            Y[i, :len(label)] = label
    
            
        inputs = {'the_input': X,
            'the_labels': Y,
            'input_length': input_length,
            'label_length': label_length,
            }
        outputs = {'ctc': np.zeros([self.batch_size])}
        return inputs, outputs

    def __one_hot(self, text,length):
        length = min(length,self.maxLabelLength)
        label = np.zeros(length)
        for i, char in enumerate(text):
            index = self.characters.find(char)
            if index == -1:
                index = self.characters.find(u'.')
            if i < length:
                label[i] = index
        return label

