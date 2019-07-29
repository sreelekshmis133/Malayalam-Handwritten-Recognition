#!/usr/bin/env python
# coding: utf-8

# In[1]:


MAL_VECTOR = 'ംഃഅആഇഈഉഊഋഌഎഏഐഒഓഔകഖഗഘങചഛജഝഞടഠഡഢണതഥദധനഩപഫബഭമയരറലളഴവശഷസഹാിീുൂൃെേൈൊോൌ്ൎൗൺൻർൽൾ.,'

# ASCII_VECTOR = '-+=!@#$%^&*(){}[]|\'"\\/?<>;:0123456789'

#ENG_VECTOR = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

CHAR_VECTOR = MAL_VECTOR#+ASCII_VECTOR

letters = [letter for letter in CHAR_VECTOR] # letter array

num_classes = len(letters) + 1               # total length of output chars + CTC separation char

img_w, img_h = 700, 32

# Network parameters
batch_size = 64
val_batch_size = 16

num = 82
img_dirpath =  (r'C:\Users\Sreelekshmi\Desktop\qwert')     

downsample_factor = 4
max_text_len = 60  # maximum text length output


# In[2]:


import cv2
import os, random
import numpy as np
#from parameter import letters,max_text_len
import os.path
             
## Input Label to Text generator
def labels_to_text(labels):   #generated labels is converted to text taking info from CHAR_VECTOR 
    return ''.join(list(map(lambda x: letters[int(x)], labels)))

def text_to_labels(text):     #label text is converted to index value taking info from CHAR_VECTOR 
    return list(map(lambda x: letters.index(x), text))

class TextImageGenerator:
    def __init__(self, img_dirpath, img_w, img_h, batch_size, downsample_factor,num , max_text_len = max_text_len):
            
        self.img_h = img_h
        self.img_w = img_w
        self.batch_size = batch_size
        self.max_text_len = max_text_len
        self.downsample_factor = downsample_factor      
        self.img_dirpath = (r'C:\Users\Sreelekshmi\Desktop\qwert')                # image dir path
        self.img_dir = os.listdir(self.img_dirpath)     # images list
        self.n = num                                    # number of images
        self.indexes = list(range(self.n))
        self.cur_index = 0
        self.imgs = np.zeros((self.n, self.img_h, self.img_w),dtype=np.int)
        self.texts = []

    def build_data(self,filename):                      # loading the entire image data into RAM, this need optimization
        print(self.n, " Image Loading start...")
       
        f = open(r'C:\Users\Sreelekshmi\Desktop\ex.txt',encoding = 'utf-8-sig')
        read = f.read()
#         print(read)
        itr = read.split('\n')
        itr = itr[0:100]
#         print(itr)
        j=0
        for line in itr:
            
#             print(line)
           
            img_file,text = line.split("-")  
#             print(text)
            if os.path.isfile(self.img_dirpath +'\\'+ img_file+ '.jpg'):
#                 print('\ntesting\n')
                img = cv2.imread(self.img_dirpath +'\\' + img_file+ ".jpg",cv2.IMREAD_GRAYSCALE)
#                 cv2.imshow("show",img)
#                 cv2.waitKey(0)
                ar = img.shape[0]/img.shape[1]
                img = cv2.resize(img,(int(self.img_h/ar), self.img_h))
                img = img.astype(np.float32)
                img = (img / 255.0) * 2.0 - 1.0            # normalizing the image to (-1-0-1) range
                if img.shape[1] <= self.img_w and len(text) <= self.max_text_len:
#                     print([len(self.texts),j])
                    self.imgs[j, :, :img.shape[1]] = img    # stores imgs 
                    img1 = cv2.convertScaleAbs(img)
                    img2 = cv2.normalize(img, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC1)
                    print(type(self.imgs[j, :, :img.shape[1]]))
#                     cv2.imshow("show",img)
#                     cv2.waitKey(0)
                    self.texts.append(text)# stores texts
                    print(len(self.texts))
                    j=j+1
                    l = self.texts
                    if len(self.texts) == self.n:
                        break   # breaks after the specified total data need to trained.
#             k=0
            
#             for i in l:
#                 k = k+1
#                 print(i)
#                 print(k)
        print(self.texts)
        print(len(self.texts))
        print(len(self.imgs))
        print(self.n)
        print(len(self.texts) == len(self.imgs))
        print(len(self.texts) == self.n)
        print(self.n, " Image Loading finish...")

    def next_sample(self):      # send one sample, increment the index to select next data 
        self.cur_index += 1
        if self.cur_index >= self.n-1:
            self.cur_index = 0
            random.shuffle(self.indexes)
        return self.imgs[self.indexes[self.cur_index]], self.texts[self.indexes[self.cur_index]]
    
c1 = TextImageGenerator(img_dirpath,img_w, img_h, batch_size, downsample_factor,num , max_text_len = max_text_len)
a = r'C:\Users\Sreelekshmi\Documents\Dataset\DB'
c1.build_data(a)
# c1.next_sample()
# print(c1.img_h)
# print(c1.img_w)
# print(c1.batch_size)
# print(c1.max_text_len)
# print(c1.downsample_factor)
# print(c1.img_dirpath)
        
#build_data(self,filename)


# In[3]:


from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda, BatchNormalization
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.layers.recurrent import LSTM, GRU
import tensorflow as tf
# from parameter import *
K.set_learning_phase(0)



# # Loss and train functions, network architecture
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def get_Model(training):
    input_shape = (img_w, img_h, 1)     # (800, 32, 1)

    # Make Networkw
    inputs = Input(name='the_input', shape=input_shape, dtype='float32')

    # Convolution layer (VGG)
    inner = Conv2D(64, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal')(inputs)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = MaxPooling2D(pool_size=(2,2), name='max1')(inner)
    #(None, 350, 16, 64)
    inner = Conv2D(128, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal')(inner)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = MaxPooling2D(pool_size=(2,2), name='max2')(inner)
    #(None, 175, 8, 128)
    inner = Conv2D(256, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal')(inner)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    #(None, 175, 8, 256)
    inner = Conv2D(256, (3, 3), padding='same', name='conv4', kernel_initializer='he_normal')(inner)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = MaxPooling2D(pool_size=(1,2), name='max3')(inner)
    #(None, 175, 4, 256)
    inner = Conv2D(512, (3, 3), padding='same', name='conv5', kernel_initializer='he_normal')(inner)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    #(None, 175, 4, 512)
    inner = Conv2D(512, (3, 3), padding='same', name='conv6')(inner)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = MaxPooling2D(pool_size=(1,2), name='max4')(inner)
    #(None, 175, 2, 512)
    inner = Conv2D(512, (2, 2), padding='same', kernel_initializer='he_normal', name='con7')(inner)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    #(None, 175, 2, 512)
    # CNN to RNN
    inner = Reshape(target_shape=((175, 1024)), name='reshape')(inner)
    #(None, 175, 1024)
    inner = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(inner)
    #(None, 175, 64) = embedding sequence fed to RNN.

    # RNN layer
    lstm_1 = LSTM(256, return_sequences=True, kernel_initializer='he_normal', name='lstm1')(inner)
    lstm_1b = LSTM(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm1_b')(inner)
    lstm1_merged = add([lstm_1, lstm_1b])
    # (None, None, 256)
    lstm1_merged = BatchNormalization()(lstm1_merged)
    lstm_2 = LSTM(256, return_sequences=True, kernel_initializer='he_normal', name='lstm2')(lstm1_merged)
    lstm_2b = LSTM(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm2_b')(lstm1_merged)
    lstm2_merged = concatenate([lstm_2, lstm_2b])
    # (None, None, 512)
    lstm_merged = BatchNormalization()(lstm2_merged)

    # transforms RNN output to character activations:
    inner = Dense(num_classes, kernel_initializer='he_normal',name='dense2')(lstm2_merged)
    #(None, 175, total_number of classes)
    y_pred = Activation('softmax', name='softmax')(inner)
    labels = Input(name='the_labels', shape=[max_text_len], dtype='float32') # (None ,8)
    input_length = Input(name='input_length', shape=[1], dtype='int64')     # (None, 1)
    label_length = Input(name='label_length', shape=[1], dtype='int64')     # (None, 1)


    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length]) #(None, 1)

    if training:
        return Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)
    else:
        return Model(inputs=[inputs], outputs=y_pred)


# In[4]:


from keras import backend as K
from keras.optimizers import Adadelta
from keras.callbacks import EarlyStopping, ModelCheckpoint
# from Image_Generator import TextImageGenerator
# from Model import get_Model
# from parameter import *
K.set_learning_phase(0)

# # Model description and training

model = get_Model(training=True)

try:
    model.load_weights('LSTM+BN4--26--0.011.hdf5')
    print("...Previous weight data...")
except:
    print("...New weight data...")
    pass

train_file_path = r'C:\Users\Sreelekshmi\Documents\Dataset\DB\train\\'
tiger_train = TextImageGenerator(train_file_path, img_w, img_h, batch_size,downsample_factor,70)   # train only first 16000 images
tiger_train.build_data(r'C:\Users\Sreelekshmi\Documents\Dataset\DB\train.csv')

valid_file_path = r'C:\Users\Sreelekshmi\Documents\Dataset\DB\test\\'
tiger_val = TextImageGenerator(valid_file_path, img_w, img_h, val_batch_size,downsample_factor,29)    # give 45 images for validation
tiger_val.build_data(r'C:\Users\Sreelekshmi\Documents\Dataset\DB\test.csv')

ada = Adadelta()

early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=4, mode='min', verbose=1)
checkpoint = ModelCheckpoint(filepath='LSTM+BN5--{epoch:02d}--{val_loss:.3f}.hdf5', monitor='loss', verbose=1, mode='min', period=1) #chechkpoint
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=ada)

print(tiger_val.n / val_batch_size)
# captures output of softmax so we can decode the output during visualization
model.fit_generator(generator=tiger_train.next_sample(),
                    steps_per_epoch=int(tiger_train.n / batch_size),
                    epochs=5, # 5 epochs
                    callbacks=[checkpoint],
                    validation_data=tiger_val.next_sample(),
                    validation_steps=int(tiger_val.n / val_batch_size))

