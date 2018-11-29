import os
import glob
import json
import imp
from keras.models import Model
from keras.layers import Input,LSTM,Dense
import numpy as np
import sys
sys.path.append('../../densenet/')
import keys_keras




with open('./similar_charater.txt','r', encoding='utf-8') as f:
    char_set = f.readlines()
char_set =[ch.strip('\n').split(' ') for ch in char_set]

characters = keys_keras.alphabet_union[:]

from collections import defaultdict
char_dict = defaultdict(str)
# char_dict = {}
for char,value in char_set:
    if (char in characters) and value in characters:
        char_dict[char] = value


# In[ ]:



with open('../create_dataset/medicine_v6.txt') as f:
    medicine = f.readlines()
medicine = [name.split('\n')[0] for name in medicine]
medicine = list(set(medicine))
medicine = [item for item in medicine if len(item) < 25]


# In[ ]:


import random

with open('./similar_charater.txt','r', encoding='utf-8') as f:
    char_set = f.readlines()
x7f = [ item for item in medicine if item.count('\x7f')]
x7f1 = keys_keras.alphabet_union.count('\x7f')
print(x7f,x7f1)


# In[ ]:


import random

with open('./similar_charater.txt','r', encoding='utf-8') as f:
    char_set = f.readlines()
char_set =[ch.strip('\n').split(' ') for ch in char_set]

characters = keys_keras.alphabet_union[:]

from collections import defaultdict
char_dict = defaultdict(str)
# char_dict = {}
for char,value in char_set:
    if (char in characters) and value in characters:
        char_dict[char] = value

def make_error_char(text):
    text_temp = text
    char = random.choice(text)
    similar_chars = char_dict[char]
    if similar_chars:
        similar_char = random.choice(similar_chars)
        text = text.replace(char,similar_char)
    return text,text_temp
def add_random_char(text):
    text_temp = text
    str_list = list(text)
    char = random.choice(characters)
    nPos = random.randint(0,len(str_list)-1)
    str_list.insert(nPos,char)
#     print(str_list)
    text = ''.join(str_list)
    return text,text_temp

def chop_random_char(text):
    text_temp = text
    char = random.choice(text)
    text = text.replace(char,'')
    return text,text_temp

def change_text(text):
    text_temp = text
    change_times = random.randint(0,4)
    functions = [add_random_char,chop_random_char]+[make_error_char]*3
    for i in range(change_times):
        function = random.choice(functions)
#         print(str(function))
        if len(text) > 1:
            text,_ = function(text)
        else:
            return text,text_temp
    return text,text_temp


# In[ ]:


medicine_total = medicine


# In[ ]:


text_label = [change_text(text) for text in medicine_total]


# In[ ]:


batch_size = 64  # Batch size for training.
epochs = 30  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.

# Vectorize the data.
input_texts = []
target_texts = []
input_characters = keys_keras.alphabet_union[:]
target_characters = input_characters

for input_text, target_text in text_label:
    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.
    target_text = target_text
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters+char
    for char in target_text:
        if char not in target_characters:
            target_characters+char
    

num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)


input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)])

encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.
print('generated Done')


# In[ ]:


# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, num_decoder_tokens),name='encoder_inputs')
encoder = LSTM(latent_dim, return_state=True,name='LSTM_encoder')
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens),name='decoder_inputs')
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True,name='LSTM_decoder')
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax',name='decoder_dense')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
# 就是不止使用了输入text的信息，还使用了label text的信息
model = Model([encoder_inputs, decoder_inputs], decoder_outputs,name='encode_decode_model')

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')


# In[ ]:


epochs=100
save_dir = './model/'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)


# In[ ]:


from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard

checkpoint = ModelCheckpoint(filepath=save_dir+'encoder_decoder-{epoch:02d}-{val_loss:.3f}-{val_acc:.3f}.h5', monitor='val_loss', save_best_only=False, save_weights_only=True)
lr_schedule = lambda epoch: 0.0002 * 0.95**epoch
learning_rate = np.array([lr_schedule(i) for i in range(epochs)])
changelr = LearningRateScheduler(lambda epoch: float(learning_rate[epoch]))
earlystop = EarlyStopping(monitor='val_acc', patience=5, verbose=1)
tensorboard = TensorBoard(log_dir='./models/logs', write_graph=True)



model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)

print(lr_schedule)
print('-----------Start training-----------')
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
    epochs = epochs,
    batch_size=batch_size,
    validation_split = 0.2,
    callbacks = [checkpoint, earlystop, changelr, tensorboard],)

print('finished training, saving model')
# Save model
model.save('seq2seq.h5_new')


# In[ ]:


# model.save_weights('seq2seq_weights_new.h5')
# model.load_weights('./seq2seq_weights_new.h5')


# # In[ ]:


# encoder_model = Model(encoder_inputs, encoder_states)

# decoder_state_input_h = Input(shape=(latent_dim,))
# decoder_state_input_c = Input(shape=(latent_dim,))
# decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
# decoder_outputs, state_h, state_c = decoder_lstm(
#     decoder_inputs, initial_state=decoder_states_inputs)
# decoder_states = [state_h, state_c]
# decoder_outputs = decoder_dense(decoder_outputs)
# decoder_model = Model(
#     [decoder_inputs] + decoder_states_inputs,
#     [decoder_outputs] + decoder_states)
# reverse_input_char_index = dict(
#     (i, char) for char, i in input_token_index.items())
# reverse_target_char_index = dict(
#     (i, char) for char, i in target_token_index.items())


# # In[ ]:


# def decode_sequence(input_seq):
#     # Encode the input as state vectors.
#     states_value = encoder_model.predict(input_seq)

#     # Generate empty target sequence of length 1.
#     target_seq = np.zeros((1, 1, num_decoder_tokens))
#     # Populate the first character of target sequence with the start character.
# #     target_seq[0, 0, target_token_index['\t']] = 1.

#     # Sampling loop for a batch of sequences
#     # (to simplify, here we assume a batch of size 1).
#     stop_condition = False
#     decoded_sentence = ''
#     while not stop_condition:
#         output_tokens, h, c = decoder_model.predict(
#             [target_seq] + states_value)

#         # Sample a token
#         sampled_token_index = np.argmax(output_tokens[0, -1, :])
#         sampled_char = reverse_target_char_index[sampled_token_index]
#         decoded_sentence += sampled_char

#         # Exit condition: either hit max length
#         # or find stop character.
#         if (sampled_char == '\n' or
#            len(decoded_sentence) > max_decoder_seq_length):
#             stop_condition = True

#         # Update the target sequence (of length 1).
#         target_seq = np.zeros((1, 1, num_decoder_tokens))
#         target_seq[0, 0, sampled_token_index] = 1.

#         # Update states
#         states_value = [h, c]

#     return decoded_sentence


# # In[ ]:


# for seq_index in range(100):
#     # Take one sequence (part of the training set)
#     # for trying out decoding.
#     input_seq = encoder_input_data[seq_index: seq_index + 1]
#     decoded_sentence = decode_sequence(input_seq)
#     print('-')
#     print('Input sentence:', input_texts[seq_index])
#     print('Decoded sentence:', decoded_sentence)


# # In[ ]:


# encoder_input_data[1:2].shape


# In[ ]:




