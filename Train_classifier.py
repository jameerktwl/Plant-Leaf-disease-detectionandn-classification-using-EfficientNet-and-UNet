# -*- coding: utf-8 -*-


from tensorflow.keras.utils import normalize
import os
from tqdm import tqdm
import numpy as np
from FeatureExtractor import extractfeatures


Classes=['Apple___Black_rot','Apple Cedar apple rust','Apple healthy','Corn_(maize) Common_rust',
         'Corn_(maize)___healthy','Corn (maize) Northern_Leaf_Blight',
         'Grape___Esca_(Black_Measles)',
        'Grape___healthy','Pepper bell Bacterial spot','Potato Early blight','Potato Late blight','Potato healthy',
       'Tomato Bacterial spot','Tomato Early blight','Tomato Late blight','Tomato Target Spot',
       'Tomato healthy']



def run():
    Segmented_directory = 'Dataset/Segmented/'## path containing segmented images
    
    Segmented = os.listdir(Segmented_directory)
    
    classes=[]
    Features=[]
    
    ###
    ###Reading all segmented  images and extracting features
    ###
    
   
    for i, folder in enumerate(Segmented): 
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
    
    np.save('Features',Features)
    np.save('classes',classes)
    
    
    Features=np.load('Features.npy')
    classes=np.load('classes.npy')
    
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(Features, classes, test_size = 0.2, random_state = 2)
    
    
    np.save('Data/X_train_classification',X_train)
    np.save('Data/X_test_classification',X_test)
    np.save('Data/y_train_classification',y_train)
    np.save('Data/y_test_classification',y_test)
    
    

    x=np.load('Data/y_test_classification.npy')
    
    
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
    
    history=model.fit(X_train,y_train,epochs=100,verbose=0,batch_size=8,validation_data=(X_test, y_test))
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
    
    
    model.save('Models/Efficient_Net')
    
    Training__accuracy=history.history['accuracy']
    validation__accuracy=history.history['val_accuracy']
    
    
    Training__loss=history.history['loss']
    validation__loss=history.history['val_loss']
    
    
    np.save('Data/Training__accuracy',Training__accuracy)
    np.save('Data/Training__loss',Training__loss)
    np.save('Data/validation__accuracy',validation__accuracy)
    np.save('Data/validation__loss',validation__loss)
    
    
    predicted=model.predict(X_test,verbose=0)
    pred = Classes[np.argmax(predicted, axis=1)[0]]
    
    np.save('Data/predicted',pred)


if __name__ == "__main__":
    run()