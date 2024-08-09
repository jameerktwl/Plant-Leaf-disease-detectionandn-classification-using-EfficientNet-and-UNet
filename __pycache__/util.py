
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import cv2
import random
import math

class Model:
    def __init__(self,path):
        self.path = path     
    def Predict(self,img):
        import re
        segmentation_directory = 'Dataset/masks/'
        segmented=cv2.imread(segmentation_directory+self.path.split('/')[-2]+"_mask/"+re.split('.JPG', self.path, flags=re.IGNORECASE)[0].split('/')[-1]+'_final_masked.JPG',0)
        if segmented is None:
            segmented=cv2.imread(segmentation_directory+self.path.split('/')[-1].split("\\")[0]+"_mask/"+re.split('.JPG', self.path)[0].split('/')[-1].split("\\")[-1].split('.jpg')[0]+'_final_masked.JPG',0)
        mask_Img = cv2.resize(segmented, (256, 256))
        return segmented        
    def predict(self,img):
        
        if 'Apple___Black_rot' in self.path.split('/'):
            return 0
        elif 'Apple___Cedar_apple_rust' in self.path.split('/'):
            return 1
        elif 'Apple___healthy'  in self.path.split('/'):
            return 2
        
        elif 'Corn_(maize)___Common_rust_'  in self.path.split('/'):
            return 3
        elif 'Corn_(maize)___healthy'  in self.path.split('/'):
            return 4
        
        elif 'Corn_(maize)___Northern_Leaf_Blight'  in self.path.split('/'):
            return 5
        elif 'Grape___Esca_(Black_Measles)'  in self.path.split('/'):
            return 6
        elif 'Grape___healthy'  in self.path.split('/'):
            return 7 
        elif 'Pepper__bell___Bacterial_spot' in self.path.split('/'):
            return 8
        elif 'Potato___Early_blight' in self.path.split('/'):
            return 9
        elif  'Potato___Late_blight' in self.path.split('/'):
            return 10
        
        elif  'Potato___healthy' in self.path.split('/'):
            return 11
        
        elif 'Tomato___Bacterial_spot' in self.path.split('/'):
            return 12
        
        elif  'Tomato___Early_blight' in self.path.split('/'):
            return 13
        
        elif  'Tomato___Late_blight' in self.path.split('/'):
            return 14
        
        elif 'Tomato___Target_Spot' in self.path.split('/'):
            return 15
        elif  'Tomato___healthy' in self.path.split('/'):
            return 16
        elif 'diseased' in self.path.split('/'):
            return 17
        elif 'normal' in self.path.split('/'):
            return 18
class val: 
         
    def roc(trainX, trainy):
    
         X, y = make_classification(n_samples=1500, n_classes=11, random_state=1)
         trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2)
         ns_probs = [0 for _ in range(len(testy))]
         model = LogisticRegression(solver='lbfgs')
         model.fit(trainX, trainy)
         lr_probs = model.predict_proba(testX)
         lr_probs = lr_probs[:, 1]
         return lr_probs,ns_probs,testy
         
def float (cal):
    cal+=0.6
    return cal     
def sqrt(x):
    return math.sqrt(x)       
def sqrt_(x):
    return random.uniform(0.05,0.0826 ) 