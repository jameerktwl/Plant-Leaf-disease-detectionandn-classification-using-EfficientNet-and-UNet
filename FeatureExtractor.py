# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
from skimage import io
import cv2
import numpy as np
from skimage.color import rgb2gray
from skimage.feature import graycoprops,SIFT,graycomatrix


def extractfeatures(path):


    patch_Size=35
    SIZE=256
    ###GLCM_Matrix
    image = cv2.imread(path)
    image = cv2.resize(image, (SIZE, SIZE))
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (SIZE, SIZE))
    
    
    ###
    ###Extracing GLCM matrix
    ###
    
    
    GLCM=graycomatrix(img,[1],[0,np.pi/4,np.pi/2,3*np.pi/4])
    props = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']
    feature = []
    glcm_props = [propery for name in props for propery in graycoprops(GLCM, name)[0]]
    
    
    def draw_cross_keypoints(img, keypoints, color):
        sift = img.copy()  
        for kp in keypoints:
            x, y = kp  
            x = int(round(x))  
            y = int(round(y))
            cv2.drawMarker(sift, (x, y), color, markerType=cv2.MARKER_CROSS, markerSize=5, thickness=1, line_type=cv2.LINE_8)
    
        return sift  
    
    
    ###
    ###Extracing SIFT Features
    ###
    
    descriptor_extractor = SIFT()
    descriptor_extractor.detect_and_extract(img)
    keypoints = descriptor_extractor.keypoints
    descriptors = descriptor_extractor.descriptors
    sift = draw_cross_keypoints(img, keypoints, color=(120,157,187)) 

    
    sigma=5
    ksize=2
    theta=1*np.pi/4
    lamda=1*np.pi/4
    gamma=1.8
    phi=2*np.pi/6
   
    
    ### 
    ###Extracing GABOUR Features
    ###
    
    
    
    kernel=cv2.getGaborKernel((ksize,ksize),sigma,theta,lamda,gamma,phi,ktype=cv2.CV_32F)
    img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gabour=cv2.filter2D(image,cv2.CV_8UC3,kernel)
    gabour=cv2.cvtColor(gabour,cv2.COLOR_BGR2GRAY)
    
    
    
    
    final=np.concatenate((gabour,sift), axis=1)
    glcm_=np.asarray(glcm_props)
    finals=final
    empty=np.zeros((1,512))
    empty[:,:glcm_.shape[0]]=glcm_
    features=np.vstack([ele for ele in [final, empty]])###Combining all these features
    
    

    return features
   

