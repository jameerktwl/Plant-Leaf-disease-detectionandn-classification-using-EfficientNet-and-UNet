# -*- coding: utf-8 -*-

import numpy as np
from Existing_models import *


Classes=['Apple Black rot','Apple Cedar apple rust','Apple healthy','Corn (maize) Common rust ',
       'Corn (maize) healthy','Corn (maize) Northern Leaf Blight','Grape Esca (Black Measles)',
       'Grape healthy',
       'Pepper bell Bacterial spot','Potato Early blight','Potato Late blight','Potato healthy',
       'Tomato Bacterial spot','Tomato Early blight','Tomato Late blight','Tomato Target Spot',
       'Tomato healthy']

Classes=np.asarray(Classes)

X_train=np.load('Data/X_train_classification.npy')
X_test=np.load('Data/X_test_classification.npy')


Y_train_=np.load('Data/y_train_classification.npy')
Y_test=np.load('Data/y_test_classification.npy')


from sklearn.preprocessing import LabelBinarizer
    

lb = LabelBinarizer()
y_train=lb.fit_transform(Y_train_)
y_test=lb.fit_transform(Y_test)

del Y_train_,Y_test

classes=np.load('classes.npy')
n_classes=len(np.unique(classes))


####Training and saving lstm
input_shape=(X_train.shape[1],X_train.shape[2])
LSTM=LS_TM(input_shape,n_classes)
LSTM.summary()
LSTM.fit(X_train,y_train,epochs=100,verbose=0,batch_size=8,validation_data=(X_test, y_test))
LSTM.save('Models/LSTM')



####Training and saving cnn
CNN=CNN(input_shape,n_classes)
CNN.summary()
CNN.fit(X_train,y_train,epochs=100,verbose=0,batch_size=8,validation_data=(X_test, y_test))
CNN.save('Models/CNN')


####Training and saving BI-LSTM
BI_LSTM=BI_LSTM(input_shape,n_classes)
BI_LSTM.summary()
BI_LSTM.fit(X_train,y_train,epochs=100,verbose=0,batch_size=8,validation_data=(X_test, y_test))
BI_LSTM.save('Models/BI_LSTM')



####Training and saving Alex-Net
Alex_Net=Alex_Net(input_shape,n_classes)
Alex_Net.summary()
Alex_Net.fit(X_train,y_train,epochs=100,verbose=0,batch_size=8,validation_data=(X_test, y_test))
Alex_Net.save('Models/Alex_Net')



####Training and saving Vgg
Vgg=Vgg(input_shape,n_classes)
Vgg.summary()
Vgg.fit(X_train,y_train,epochs=100,verbose=0,batch_size=8,validation_data=(X_test, y_test))
Vgg.save('Models/Vgg')



####Training and saving Resnet
Resnet=ResNet34(input_shape,n_classes)
Resnet.summary()
Resnet.fit(X_train,y_train,epochs=100,verbose=0,batch_size=8,validation_data=(X_test, y_test))
Resnet.save('Models/Resnet')


####Testing LSTM
predicted=LSTM.predict(X_test,verbose=0)
pred = Classes[np.argmax(predicted, axis=1)[0]]
np.save('Data/predicted_lstm',pred)


####Testing CNN
predicted=CNN.predict(X_test,verbose=0)
pred = Classes[np.argmax(predicted, axis=1)[0]]
np.save('Data/predicted_cnn',pred)


####Testing Bi-LSTM
predicted=BI_LSTM.predict(X_test,verbose=0)
pred = Classes[np.argmax(predicted, axis=1)[0]]
np.save('Data/predicted_bilstm',pred)

####Testing Alex-Net
predicted=Alex_Net.predict(X_test,verbose=0)
pred = Classes[np.argmax(predicted, axis=1)[0]]
np.save('Data/predicted_alex',pred)

####Testing Vgg
predicted=Vgg.predict(X_test,verbose=0)
pred = Classes[np.argmax(predicted, axis=1)[0]]
np.save('Data/predicted_vgg',pred)

####Testing Resnet
predicted=Resnet.predict(X_test,verbose=0)
pred = Classes[np.argmax(predicted, axis=1)[0]]
np.save('Data/predicted_resnet',pred)
