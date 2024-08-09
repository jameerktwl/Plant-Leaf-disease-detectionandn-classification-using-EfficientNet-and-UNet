# -*- coding: utf-8 -*-


from tensorflow.keras.utils import normalize
import os
from tqdm import tqdm
import numpy as np
from FeatureExtractor import extractfeatures


Classes=['Healthy','Diseased']



def run():
    Segmented_directory = 'Dataset/Segmented/'## path containing segmented images
    
    Segmented = os.listdir(Segmented_directory)
    
    classes=[]
    Features=[]
    
    ###
    ###Reading all segmented  images and extracting features
    ###
    
    no=['normal','diseased']
    for i, folder in enumerate(Segmented): 
        print(folder)
        
        if folder in no:
            for itr, image in enumerate(os.listdir(os.path.join(Segmented_directory,folder))): 
                path=os.path.join(Segmented_directory+folder,image)
                features=extractfeatures(path) 
                features=features.flatten()
                features=features.reshape(int(features.shape[0]/2),2)
                features=features.astype('uint16')
                classes.append(folder)
                Features.append(features)
                if itr==200:
                    break
        
    Features=np.asarray(Features)
    classes=np.asarray(classes)
    
    np.save('Features_Clients_Data',Features)
    np.save('classes_Clients_Data',classes)
    
    
    Features=np.load('Features_clients_data.npy')
    classes=np.load('classes_clients_data.npy')
    
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(Features, classes, test_size = 0.2, random_state = 2)
    
    
    np.save('Data/X_train_classification_Clients_Data',X_train)
    np.save('Data/X_test_classification_Clients_Data',X_test)
    np.save('Data/y_train_classification_Clients_Data',y_train)
    np.save('Data/y_test_classification_Clients_Data',y_test)
    
   
    
    
    from sklearn.preprocessing import LabelBinarizer
    
    
    lb = LabelBinarizer()
    y_train=lb.fit_transform(y_train)
    y_test=lb.fit_transform(y_test)
    
    classnum=len(np.unique(classes))
    
    
    
    ##
    ## Initializing our proposed model
    ##
    
    print('')
    
    
    from Model import EffNet
    model=EffNet((X_train.shape[1],X_train.shape[2]),classnum,plot_model=False)   
    
    
    
    ####
    #### Training of our proposed model
    ####
    
    history=model.fit(X_train,y_train,epochs=1,verbose=0,batch_size=8,validation_data=(X_test, y_test))
    weights=model.get_weights()
    
    
    
    #####
    ##### Hybrid_DTBO  for weight optimization 
    #####
    
    
    from Hybrid_DTBO import main
    BestX=main.Hybrid_DTBO(weights)
    model.set_weights(weights)
    
    
    
    ####
    #### Saving  our proposed model after training
    #### 
    model.save('Models/Efficient_Net_clients_Dataset')

    
    predicted=model.predict(X_test,verbose=0)
    pred = Classes[np.argmax(predicted, axis=1)[0]]
    
    np.save('Data/predicted_clients_data',pred)
    
    input_shape=(X_train.shape[1],X_train.shape[2])
    LSTM=LS_TM(input_shape,n_classes)
    LSTM.summary()
    LSTM.fit(X_train,y_train,epochs=1,verbose=1,batch_size=8,validation_data=(X_test, y_test))
    model.save('Models/LSTM_clients_Dataset')
    
    
    CNN=CNN(input_shape,n_classes)
    CNN.summary()
    CNN.fit(X_train,y_train,epochs=1,verbose=1,batch_size=8,validation_data=(X_test, y_test))
    model.save('Models/CNN_clients_Dataset')
    
    
    
    BI_LSTM=BI_LSTM(input_shape,n_classes)
    BI_LSTM.summary()
    BI_LSTM.fit(X_train,y_train,epochs=1,verbose=1,batch_size=8,validation_data=(X_test, y_test))
    model.save('Models/BiLSTM_clients_Dataset')
    
    
    Alex_Net=Alex_Net(input_shape,n_classes)
    Alex_Net.summary()
    Alex_Net.fit(X_train,y_train,epochs=1,verbose=1,batch_size=8,validation_data=(X_test, y_test))
    model.save('Models/Alexnet_clients_Dataset')
    
    Vgg=Vgg(input_shape,n_classes)
    Vgg.summary()
    Vgg.fit(X_train,y_train,epochs=1,verbose=1,batch_size=8,validation_data=(X_test, y_test))
    model.save('Models/Vgg_clients_Dataset')
    
    
    Resnet=ResNet34(input_shape,n_classes)
    Resnet.summary()
    Resnet.fit(X_train,y_train,epochs=1,verbose=1,batch_size=8,validation_data=(X_test, y_test))
    model.save('Models/Resnet_clients_Dataset')    
    
    
    
    






















if __name__ == "__main__":
    run()
