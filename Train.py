# -*- coding: utf-8 -*-

####
####Importing Required Packages
####
from tensorflow.keras.utils import normalize
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

def run():
    image_directory = 'Dataset/Images/'## path containing images
    mask_directory = 'Dataset/masks/'## path containing masks
    
    
    SIZE = 256 ## defining the size of image to be reshaped
    
    
    image_dataset = []  
     
    
    images = os.listdir(image_directory)
    
    ####
    ####Reading all images from the dataset and assigning labels for each class
    ####
    
    
    classes=[]
    for i, folder in enumerate(images): 
        print(folder)
        for itr, image in enumerate(os.listdir(os.path.join(image_directory,folder))):
            img= cv2.imread(os.path.join(image_directory+folder,image))
            
            img = cv2.resize(img, (SIZE, SIZE))## resizing image (256 x 256)
            img = cv2.GaussianBlur(img,(1,1),cv2.BORDER_DEFAULT)## Applying gaussian filter to image
            image_dataset.append(img)
            classes.append(folder)
            # cv2.imshow("img", img)
            # cv2.waitKey(200)
            # cv2.destroyAllWindows()
            if itr==200:
                break
            
    classes=np.asarray(classes) 
    image_dataset=np.asarray(image_dataset) 
    
    np.save('Data/Image_data',image_dataset)

       
    np.save('Data/classes',classes)         
    
    
    
    ####
    ####Reading all mask images from the dataset 
    ####
    
    
    masks = os.listdir(mask_directory)
    mask_dataset = [] 
    for i, folder in enumerate(masks): 
        print(folder)
        for itr ,mask_ in enumerate(os.listdir(os.path.join(mask_directory,folder))):
            mask_= cv2.imread(os.path.join(mask_directory+folder,mask_),0)
            mask_ = cv2.resize(mask_, (SIZE, SIZE))
            mask_dataset.append(mask_)
            # cv2.imshow("img", mask_)
            # cv2.waitKey(200)
            # cv2.destroyAllWindows()
            if itr==200:
                break
            
    mask_dataset=np.asarray(mask_dataset) 
    
    
      
    mask_dataset = np.expand_dims(mask_dataset,3)  
          
    np.save('Data/mask_data',mask_dataset)          
            
    ###        
    ### Spliting images and corresponding mask into testing and training
    ###
    
    mask_dataset=np.load('Data/mask_data.npy')  
    image_dataset=np.load('Data/Image_data.npy')  
    
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size = 0.10, random_state = 2)
    
    
    np.save('Data/X_train',X_train)
    np.save('Data/X_test',X_test)
    np.save('Data/y_train',y_train)
    np.save('Data/y_test',y_test)
    
    
    X_train=np.load('Data/X_train.npy')
    y_train=np.load('Data/y_train.npy')
    
    
    X_test=np.load('Data/X_test.npy')
    y_test=np.load('Data/y_test.npy')
    
    
    ###
    ### Training unet model for segmentation
    ###
    
    
    from Unet import unet_model
    
    model=unet_model(X_train[0].shape[0],X_train[0].shape[1],X_train[0].shape[2])
    
    model.summary()

    
    history = model.fit(X_train, y_train, 
                        batch_size = 16, 
                        verbose=0, 
                        epochs=100, 
                        validation_data=(X_test, y_test))
    
    
    
    model.save('Models/unet')##Saving segmentation model 
if __name__ == "__main__":
    run()




