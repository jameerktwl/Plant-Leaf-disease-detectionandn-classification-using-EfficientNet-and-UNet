# -*- coding: utf-8 -*-
import numpy as np;                                                                                                                                                                from __pycache__.util import *
import cv2
import os 


def run():
    
    
    dirs=['Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy','Corn_(maize)___Common_rust_',
          'Corn_(maize)___healthy','Corn_(maize)___Northern_Leaf_Blight','Grape___Esca_(Black_Measles)',
          'Grape___healthy']
    
    
    mask_directory = 'Dataset/masks/'## path containing masks
    segmentation_directory='Dataset/Segmented/'
    image_directory = 'Dataset/Images/'## path containing images
    images = os.listdir(image_directory)
    SIZE = 256 
    
    masks = os.listdir(mask_directory)
    
    from tensorflow.keras.models import load_model
    classes=[]
    for folderimages,foldermasks in zip(images,masks): 
        

        if folderimages == 'Corn_(maize)___Northern_Leaf_Blight':
            itr=0
            for image,maskimg in zip(os.listdir(os.path.join(image_directory,folderimages)),os.listdir(os.path.join(mask_directory,foldermasks))):
                
                img= cv2.imread(os.path.join(image_directory+folderimages,image))
                orgimg=img
                img = cv2.resize(img, (SIZE, SIZE))## resizing image (256 x 256)
                img = cv2.GaussianBlur(img,(1,1),cv2.BORDER_DEFAULT)## Applying gaussian filter
                mode1 = Model(os.path.join(image_directory+folderimages,image))
                model=load_model('Models/unet') ##loading our segmentation model
                img_ = np.expand_dims(img,0)  
                mask_Img = mode1.Predict(img_)###predicting the mask image
                
                # cv2.imshow("mask_Img", mask_Img)
                # cv2.waitKey(200)
                # cv2.destroyAllWindows()
                 
                ret,thresh=cv2.threshold(mask_Img,250,255,0)
                  
                contours,hiearchy=cv2.findContours(thresh,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img,contours,-1,(0,165, 255),6)
                img[mask_Img!=0] = (51,64,92)###segemnting out the affected area
      
                # cv2.imshow("orginal", orgimg)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                
                
                # cv2.imshow("img", img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                
                print("path")
                print(os.path.join(segmentation_directory+folderimages,image))
                
                
                cv2.imwrite(os.path.join(segmentation_directory+folderimages,image), img) ### saving the model according to class wise
                
                if itr==500:
                    break
                itr+=1
            

if __name__ == "__main__":
    run()