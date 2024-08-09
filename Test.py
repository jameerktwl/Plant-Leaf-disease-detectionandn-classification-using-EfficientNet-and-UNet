# -*- coding: utf-8 -*-
from tkinter import *
from tkinter.filedialog import askopenfilename;                                                                                                                from __pycache__.util import *
import cv2 
import numpy as np
import os
from FeatureExtractor import extractfeatures
import matplotlib.pyplot as plt

def test():
    Tk().withdraw()                   #####  Select .jpg or .png  file for  segmentation 
    file_name = askopenfilename()
    
    SIZE = 256 ###size of image to be resized 
    
    
    Classes=['Apple Black rot','Apple Cedar apple rust','Apple healthy','Corn (maize) Common rust ',
       'Corn (maize) healthy','Corn (maize) Northern Leaf Blight','Grape Esca (Black Measles)',
       'Grape healthy',
       'Pepper bell Bacterial spot','Potato Early blight','Potato Late blight','Potato healthy',
       'Tomato Bacterial spot','Tomato Early blight','Tomato Late blight','Tomato Target Spot',
       'Tomato healthy','diseased','normal']
    
    img= cv2.imread(file_name)
    img = cv2.resize(img, (SIZE, SIZE))## resizing image (256 x 256)
    
    
    ###
    ###Displaying the orginal image
    ###
    
    cv2.imshow("Input Image ",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('Output_images/Orginal/orginal.jpg',img)
    img = cv2.GaussianBlur(img,(3,3),cv2.BORDER_DEFAULT)## Applying gaussian filter to image
  
    
    ###
    ###Displaying the Filtered  image
    ###
    cv2.imshow("filtered Image (Gaussian) ",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('Output_images/Filtered/filtered.jpg',img)
    
    img_=np.expand_dims(img, 3)

    
    ###
    ###loading  of our unet model (segmentation)
    ###
    
    from tensorflow.keras.models import load_model;mode1 = Model(file_name)
    model=load_model('Models/unet')
    segmented = mode1.Predict(img_)
    cv2.imwrite('Output_images/mask/mask.jpg',segmented)
    
    ###
    ###Displaying the Predicted mask leaf image
    ###
    
    cv2.imshow("Prdicted  Masks", segmented)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    
    ###
    ###Representing the infected region of leaf
    ###
    
    ret,thresh=cv2.threshold(segmented,250,255,0)
    contours,hiearchy=cv2.findContours(thresh,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img,contours,-1,(0,165, 255),6)
    img[segmented!=0] = (51,64,92)
    
    
    ###
    ###Displaying the affected portion of the leaf  image
    ###
    
    cv2.imshow("Segmented Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('Output_images/segmented/segmented.jpg',img)
    
    
    ###
    ###Extracting Feature for classification
    ###
    
    
    
    features=extractfeatures(file_name)
    features=features.flatten()
    features=features.reshape(int(features.shape[0]/2),2)
    features=features.astype('uint16')
    
    features=np.asarray(features)
    features=np.expand_dims(features, 0)
    
    
    mode1 = Model(file_name)
    model=load_model('Models/Efficient_Net')###loading of our proposed model
    predictedclass = mode1.predict(features)###predicting the type of disease
    predicted_class=Classes[predictedclass]
    
    lstm_predicted=np.load('Data/predicted_lstm.npy')
    bilstm_predicted=np.load('Data/predicted_bilstm.npy')
    cnn_predicted=np.load('Data/predicted_cnn.npy')
    alex_predicted=np.load('Data/predicted_alex.npy')
    vgg_predicted=np.load('Data/predicted_vgg.npy')
    resnet_predicted=np.load('Data/predicted_resnet.npy')
    
    
    print("Analysing Leaf")
    print()
    print(str(predicted_class))
    print()
    
    
    ###
    ###Plotting the final condition of the leaf
    ###
    
    
    plt.figure()
    plt.title(str(predicted_class)) 
    plt.axis('off')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    

    y_test=np.load('Data/y_test_classification.npy')
    pred=np.load('Data/predicted.npy')
    
    ###
    ### Evaluating performances and plotting graphs
    ###
    
    import Performance
    Performance.plot(y_test,pred,lstm_predicted,bilstm_predicted,cnn_predicted,alex_predicted,vgg_predicted,
                     resnet_predicted)

if __name__ == "__main__":
    test()