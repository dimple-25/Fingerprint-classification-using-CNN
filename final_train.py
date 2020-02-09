from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import adadelta,SGD
from keras.utils import np_utils

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import theano
from PIL import Image
from numpy import *

img_rows, img_cols = 200,200
img_channels = 1
path1 = '/content/drive/My Drive/Colab Notebooks/New folder'
path2 = '/content/drive/My Drive/Colab Notebooks/final_dataset__1'

listing = os.listdir(path1)
num_samples=size(listing)
print(num_samples)

for file in listing:
    im = Image.open(path1 + '\\' + file)  
    img = im.resize((img_rows,img_cols))
    gray = img.convert('L')        
    gray.save(path2 +'\\' +  file, "jpeg")

imlist = os.listdir(path2)
im1 = array(Image.open('/content/drive/My Drive/Colab Notebooks/final_dataset__1' + '//'+ imlist[0])) 
m,n = im1.shape[0:2] 
imnbr = len(imlist) 
immatrix = array([array(Image.open('/content/drive/My Drive/Colab Notebooks/final_dataset__1'+ '//' + im2)).flatten()
              for im2 in imlist],'f')

label=np.ones((num_samples,),dtype = int)
label[0:218]=1
label[218:601]=0
label[601:885]=2
label[885:1054]=3
label[1054:1352]=4

data,label = shuffle(immatrix,label, random_state=2)
train_data = [data,label]
img=immatrix[2].reshape(img_rows,img_cols)
plt.imshow(img,cmap='gray')

print (train_data[0].shape)
print (train_data[1].shape)

(X, y) = (train_data[0],train_data[1])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols,1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols,1)
print('X_train shape:', X_train.shape)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

Y_train = np_utils.to_categorical(y_train, 5)
Y_test = np_utils.to_categorical(y_test, 5)

num_classes = 5
input_shape = 200,200,1

model = Sequential()

model.add(Conv2D(48, (11,11),border_mode='same',input_shape=(img_rows, img_cols,1)))
convout1 = Activation('relu')
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(convout1)
model.add(Conv2D(256,(5,5)))
convout2 = Activation('relu')
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(convout2)
model.add(Conv2D(192,(3,3)))
convout3 = Activation('relu')
model.add(convout3)
model.add(Conv2D(128,(3,3)))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adadelta',metrics=["accuracy"])

hist = model.fit(X_train, Y_train, batch_size=32, nb_epoch=num_epoch,
              verbose=1, validation_split=0.2)

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(30)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
print(plt.style.available) 
plt.style.use(['classic'])

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
print (plt.style.available) 
plt.style.use(['classic'])

from sklearn.metrics import classification_report,confusion_matrix

Y_pred = model.predict(X_test)
print(Y_pred)
y_pred = np.argmax(Y_pred, axis=1)
print(y_pred)

y_pred = model.predict_classes(X_test)
print(y_pred)

p=model.predict_proba(X_test) 
p

target_names = ['class 0(arch)', 'class 1(left)', 'class 2(right)', 'class 3(tented)', 'class 4(whorl)']
print(classification_report(np.argmax(Y_test,axis=1), y_pred,target_names=target_names))
print(confusion_matrix(np.argmax(Y_test,axis=1), y_pred))

import cv2
import tensorflow as tf

model.save('cnn2.h5')
model = tf.keras.models.load_model('/content/cnn1.h5')

test_image = cv2.imread('/content/drive/My Drive/Colab Notebooks/for_prediction/left2.jpeg')
test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
test_image = cv2.resize(test_image,(200,200))
test_image = np.array(test_image)
test_image = test_image.astype('float32')
test_image /= 255
print (test_image.shape)

from keras import backend as K

if K.image_data_format()=='channel_first':
		test_image= np.expand_dims(test_image, axis=0)
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)
else:
		test_image= np.expand_dims(test_image, axis=3) 
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)

print((model.predict(test_image)))
print(model.predict_classes(test_image))

