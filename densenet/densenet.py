from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation, Reshape, Permute
from keras.layers.convolutional import Conv2D, Conv2DTranspose, ZeroPadding2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D
from keras.layers import Input, Flatten
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.layers.wrappers import TimeDistributed


def conv_block(input, growth_rate, dropout_rate=None, weight_decay=1e-4,name=None):
    x = BatchNormalization(axis=-1, epsilon=1.1e-5,name=name+'_0_bn')(input)
    x = Activation('relu',name=name+'_0_relu')(x)
    x = Conv2D(growth_rate, (3,3), kernel_initializer='he_normal', padding='same',name=name+'_1_conv')(x)
    if(dropout_rate):
        x = Dropout(dropout_rate,name=name+'_1_dropout')(x)
    return x

def dense_block(x, nb_layers, nb_filter, growth_rate, droput_rate=0.2, weight_decay=1e-4,name=None):
    for i in range(nb_layers):
        cb = conv_block(x, growth_rate, droput_rate, weight_decay,name=name + '_block' + str(i + 1))
        x = concatenate([x, cb], axis=-1,name=name + '_concat'+ str(i + 1))
        nb_filter += growth_rate
    return x, nb_filter

def transition_block(input, nb_filter,dropout_rate=None, pooltype=1, weight_decay=1e-4,name=None):
    x = BatchNormalization(axis=-1, epsilon=1.1e-5,name=name + '_bn')(input)
    x = Activation('relu',name=name + '_relu')(x)
    x = Conv2D(nb_filter, (1, 1), kernel_initializer='he_normal', padding='same', use_bias=False,
               kernel_regularizer=l2(weight_decay),name=name + '_conv')(x)

    if(dropout_rate):
        x = Dropout(dropout_rate,name=name + '_dropout')(x)

    if(pooltype == 2):
        x = AveragePooling2D((2, 2), strides=(2, 2),name=name + '_0_avgpool')(x)
    elif(pooltype == 1):
        x = ZeroPadding2D(padding = (0, 1),name=name + '_0_zeropool')(x)
        x = AveragePooling2D((2, 2), strides=(2, 1),name=name + '_1_avgpool')(x)
    elif(pooltype == 3):
        x = AveragePooling2D((2, 2), strides=(2, 1),name=name + '_2_avgpool')(x)
    return x, nb_filter

def dense_cnn(input, nclass):

    _dropout_rate = 0.2 
    _weight_decay = 1e-4

    _nb_filter = 64
    # conv 64 5*5 s=2
    x = Conv2D(_nb_filter, (5, 5), strides=(2, 2), kernel_initializer='he_normal', padding='same',
               use_bias=False, kernel_regularizer=l2(_weight_decay),name='conv1/conv')(input)
   
    # 64 + 8 * 8 = 128
    x, _nb_filter = dense_block(x, 8, _nb_filter, 8, None, _weight_decay,name='conv2',)
    # 128
    x, _nb_filter = transition_block(x, 128, _dropout_rate, 2, _weight_decay,name='pool2')

    # 128 + 8 * 8 = 192
    x, _nb_filter = dense_block(x, 8, _nb_filter, 8, None, _weight_decay,name='conv3')
    # 192 -> 128
    x, _nb_filter = transition_block(x, 128, _dropout_rate, 2, _weight_decay,name='pool3')

    # 128 + 8 * 8 = 192
    x, _nb_filter = dense_block(x, 8, _nb_filter, 8, None, _weight_decay,name='conv4')

    x = BatchNormalization(axis=-1, epsilon=1.1e-5,name='bn')(x)
    x = Activation('relu',name='relu')(x)

    x = Permute((2, 1, 3), name='permute')(x)
    x = TimeDistributed(Flatten(), name='flatten')(x)
    y_pred = Dense(nclass, name='out', activation='softmax')(x)

    # basemodel = Model(inputs=input, outputs=y_pred)
    # basemodel.summary()

    return y_pred

def dense_blstm(input):

    pass

input = Input(shape=(32, 280, 1), name='the_input')
dense_cnn(input, 5000)
