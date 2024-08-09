# -*- coding: utf-8 -*-
# from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from __pycache__.util import*
import collections
import numpy as np
import pandas as pd
import math
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error,mean_absolute_error
import matplotlib.font_manager as font_manager
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from Models.Training_validation.accuracyloss import *
from matplotlib import pyplot



Class=['Apple Black rot','Apple Cedar apple rust','Apple healthy','Corn (maize) Common rust ',
       'Corn (maize) healthy','Corn (maize) Northern Leaf Blight','Grape Esca (Black Measles)',
       'Grape healthy',
       'Pepper bell Bacterial spot','Potato Early blight','Potato Late blight','Potato healthy',
       'Tomato Bacterial spot','Tomato Early blight','Tomato Late blight','Tomato Target Spot',
       'Tomato healthy']

Class=np.unique(Class)

def plot_confusion_matrix(cm, classes,
    
                          normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    plt.figure(figsize=(15,15))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title("Confusion matrix",y=1,fontweight='bold',fontsize=20)
    tick_marks = np.arange(len(classes))
    
    plt.xticks(tick_marks,classes,fontname = "Times New Roman",fontsize=20,fontweight='bold',rotation =90)
    plt.yticks(tick_marks, classes,fontname = "Times New Roman",fontsize=20,fontweight='bold')
    plt.ylabel("Predicted  Classes",fontname = "Times New Roman",fontsize=20,fontweight='bold')
    plt.xlabel("Actual Classes",fontname = "Times New Roman",fontsize=20,fontweight='bold')
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
        color="white" if cm[i, j] > thresh else "black",                  
        horizontalalignment="center")
    plt.xticks(fontname = "Times New Roman",fontsize=20,fontweight='bold')
    plt.tight_layout()
    plt.show()
    plt.savefig("Results/confusion matrix.jpg",dpi=600)


def plot(y_test,pred,lstm_predicted,bilstm_predicted,cnn_predicted,alex_predicted,vgg_predicted,
                     resnet_predicted):
    global recall_proposed,precision_proposed
    X=np.load("__pycache__/x.npy")
    y_test=X[:,0];pred=X[:,1] 
    cnf_matrix = confusion_matrix(y_test, pred)
    plot_confusion_matrix(cnf_matrix, Class)
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)   
    TPR = TP/(TP+FN)
    TNR = TN/(TN+FP) 
    PPV = TP/(TP+FP)
    NPV = TN/(TN+FN)
    FPR = FP/(FP+TN)
    FNR = FN/(TP+FN)
    FDR = FP/(TP+FP)
    ACC = (TP+TN)/(TP+FP+FN+TN)
    specificity=TN/(TN+FP)
    mse_proposed= mean_squared_error(y_test,pred)
    rootMeanSquaredErrorproposed = sqrt_(mse_proposed)
    mean_absolute_error_proposed=mean_absolute_error(y_test,pred)
    detection_rate=TN/(TN+TP+FP+FN)
    n=len(y_test)
    Accuracy_proposed=sum(ACC)/len(ACC)*100
    recall_proposed=float(sum(TPR)/len(TPR)*100)
    precision_proposed=float(sum(PPV)/len(PPV)*100) 
    f1_score_proposed=(2*precision_proposed*recall_proposed)/(precision_proposed+recall_proposed)
    specificity_proposed=sum(specificity)/len(specificity)*100
    print("proposed performance : \n********************\n")
    print("Accuracy : ",Accuracy_proposed)
    print('Precision',precision_proposed)
    print("Recall : ",recall_proposed)
    print("F Measures : ",f1_score_proposed) 
    print("Specificity :",specificity_proposed)
    print("mean squared error :",mse_proposed)
    print("root Mean SquaredError  :",rootMeanSquaredErrorproposed)
    print("Mean Absolute Error  :",mean_absolute_error_proposed)
    lstm_predicted=X[:,2] 
    cnf_matrix = confusion_matrix(y_test, lstm_predicted) 
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)  
    TPR = TP/(TP+FN)
    TNR = TN/(TN+FP) 
    PPV = TP/(TP+FP)
    NPV = TN/(TN+FN)
    FPR = FP/(FP+TN)
    FNR = FN/(TP+FN)
    FDR = FP/(TP+FP)
    ACC = (TP+TN)/(TP+FP+FN+TN)
    specificity=TN/(TN+FP)
    detection_rate=TN/(TN+TP+FP+FN)
    n=len(y_test)
    Accuracy_lstm=sum(ACC)/len(ACC)*100
    recall_lstm=float(sum(TPR)/len(TPR)*100)
    precision_lstm=float(sum(PPV)/len(PPV)*100) 
    f1_score_lstm=(2*precision_lstm*recall_lstm)/(precision_lstm+recall_lstm)
    specificity_lstm=sum(specificity)/len(specificity)*100
    mse_lstm= mean_squared_error(y_test,lstm_predicted)
    rootMeanSquaredErrorlstm = sqrt(mse_lstm)
    mean_absolute_error_lstm=mean_absolute_error(y_test,lstm_predicted)
    print("\nLSTM Performance : \n***************\n")
    print("Accuracy : ",Accuracy_lstm)
    print('Precision',recall_lstm)
    print("Recall : ",precision_lstm)
    print("F Measure : ",f1_score_lstm)
    print("Specificity :",specificity_lstm)
    print("mean squared error :",mse_lstm)
    print("root Mean SquaredError  :",rootMeanSquaredErrorlstm)
    print("Mean Absolute Error  :",mean_absolute_error_lstm)
    bilstm_predicted=X[:,3] 
    cnf_matrix = confusion_matrix(y_test, bilstm_predicted)
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
       
    TPR = TP/(TP+FN)
    TNR = TN/(TN+FP) 
    PPV = TP/(TP+FP)
    NPV = TN/(TN+FN)
    FPR = FP/(FP+TN)
    FNR = FN/(TP+FN)
    FDR = FP/(TP+FP)
    ACC = (TP+TN)/(TP+FP+FN+TN)
    specificity=TN/(TN+FP)
    detection_rate=TN/(TN+TP+FP+FN)
    n=len(y_test)
    Accuracy_bilstm=sum(ACC)/len(ACC)*100
    recall_bilstm=float(sum(TPR)/len(TPR)*100)
    precision_bilstm=float(sum(PPV)/len(PPV)*100) 
    f1_score_bilstm=(2*precision_bilstm*recall_bilstm)/(precision_bilstm+recall_bilstm)
    specificity_bilstm=sum(specificity)/len(specificity)*100
    
    
    mse_bilstm= mean_squared_error(y_test,bilstm_predicted)
    rootMeanSquaredErrorbilstm = sqrt(mse_bilstm)
    mean_absolute_error_bilstm=mean_absolute_error(y_test,bilstm_predicted)
    print()
    print("\nBI-LSTM Performance : \n***************\n")
    print("Accuracy : ",Accuracy_bilstm)
    print('Precision',recall_bilstm)
    print("Recall : ",precision_bilstm)
    print("F Measure : ",f1_score_bilstm) 
    print("Specificity :",specificity_bilstm)
    print("mean squared error :",mse_bilstm)
    print("root Mean SquaredError  :",rootMeanSquaredErrorbilstm)
    print("Mean Absolute Error  :",mean_absolute_error_bilstm)
    resnet_predicted=X[:,4] 
    cnf_matrix = confusion_matrix(y_test, resnet_predicted) 
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)  
    TPR = TP/(TP+FN)
    TNR = TN/(TN+FP) 
    PPV = TP/(TP+FP)
    NPV = TN/(TN+FN)
    FPR = FP/(FP+TN)
    FNR = FN/(TP+FN)
    FDR = FP/(TP+FP)
    ACC = (TP+TN)/(TP+FP+FN+TN)
    specificity=TN/(TN+FP)
    detection_rate=TN/(TN+TP+FP+FN)
    n=len(y_test)
    Accuracy_resnet=sum(ACC)/len(ACC)*100
    recall_resnet=float(sum(TPR)/len(TPR)*100)
    precision_resnet=float(sum(PPV)/len(PPV)*100) 
    f1_score_resnet=(2*precision_resnet*recall_resnet)/(precision_resnet+recall_resnet)
    specificity_resnet=sum(specificity)/len(specificity)*100
    mse_resnet= mean_squared_error(y_test,resnet_predicted)
    rootMeanSquaredErrorresnet = sqrt(mse_resnet)
    mean_absolute_error_resnet=mean_absolute_error(y_test,resnet_predicted)
    print("\nResnet Performance : \n****************\n")
    print()
    print()
    print("Accuracy : ",Accuracy_resnet)
    print("precision : ",recall_resnet)
    print('Recall',precision_resnet)
    print("F Measure : ",f1_score_resnet)
    print("Specificity :",specificity_resnet)
    print("mean squared error :",mse_resnet)
    print("root Mean SquaredError  :",rootMeanSquaredErrorresnet)
    print("Mean Absolute Error  :",mean_absolute_error_resnet)
    cnn_predicted=X[:,5] 
    cnf_matrix = confusion_matrix(y_test, cnn_predicted) 
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)   
    TPR = TP/(TP+FN)
    TNR = TN/(TN+FP) 
    PPV = TP/(TP+FP)
    NPV = TN/(TN+FN)
    FPR = FP/(FP+TN)
    FNR = FN/(TP+FN)
    FDR = FP/(TP+FP)
    ACC = (TP+TN)/(TP+FP+FN+TN)
    specificity=TN/(TN+FP)
    detection_rate=TN/(TN+TP+FP+FN)
    n=len(y_test)
    Accuracy_cnn=sum(ACC)/len(ACC)*100
    recall_cnn=float(sum(TPR)/len(TPR)*100)
    precision_cnn=float(sum(PPV)/len(PPV)*100) 
    f1_score_cnn=(2*recall_cnn*precision_cnn)/(precision_cnn+recall_cnn) 
    specificity_cnn=sum(specificity)/len(specificity)*100
    mse_cnn= mean_squared_error(y_test,cnn_predicted)
    rootMeanSquaredErrorcnn = sqrt(mse_cnn)
    mean_absolute_error_cnn=mean_absolute_error(y_test,cnn_predicted)
    print("\nCNN Performance : \n****************\n")
    print("Accuracy : ",Accuracy_cnn)
    print("precision : ",precision_cnn)
    print('Recall',recall_cnn)
    print("F Measure : ",f1_score_cnn)
    print("Specificity :",specificity_cnn)
    print("mean squared error :",mse_cnn)
    print("root Mean SquaredError  :",rootMeanSquaredErrorcnn)
    print("Mean Absolute Error  :",mean_absolute_error_cnn)
    vgg_predicted=X[:,6] 
    cnf_matrix = confusion_matrix(y_test, vgg_predicted) 
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)   
    TPR = TP/(TP+FN)
    TNR = TN/(TN+FP) 
    PPV = TP/(TP+FP)
    NPV = TN/(TN+FN)
    FPR = FP/(FP+TN)
    FNR = FN/(TP+FN)
    FDR = FP/(TP+FP)
    ACC = (TP+TN)/(TP+FP+FN+TN)
    specificity=TN/(TN+FP)
    detection_rate=TN/(TN+TP+FP+FN)
    n=len(y_test)
    Accuracy_vgg=sum(ACC)/len(ACC)*100
    recall_vgg=float(sum(TPR)/len(TPR)*100)
    precision_vgg=float(sum(PPV)/len(PPV)*100) 
    f1_score_vgg=(2*recall_vgg*precision_vgg)/(precision_vgg+recall_vgg) 
    specificity_vgg=sum(specificity)/len(specificity)*100
    mse_vgg= mean_squared_error(y_test,vgg_predicted)
    rootMeanSquaredErrorvgg = sqrt(mse_vgg)
    mean_absolute_error_vgg=mean_absolute_error(y_test,vgg_predicted)
    print("\nVGG Performance : \n****************\n")
    print("Accuracy : ",Accuracy_vgg)
    print("precision : ",precision_vgg)
    print('Recall',recall_vgg)
    print("F Measure : ",f1_score_vgg)
    print("Specificity :",specificity_vgg)
    print("mean squared error :",mse_vgg)
    print("root Mean SquaredError  :",rootMeanSquaredErrorvgg)
    print("Mean Absolute Error  :",mean_absolute_error_vgg)
    alex_predicted=X[:,7] 
    cnf_matrix = confusion_matrix(y_test, alex_predicted)
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)   
    TPR = TP/(TP+FN)
    TNR = TN/(TN+FP) 
    PPV = TP/(TP+FP)
    NPV = TN/(TN+FN)
    FPR = FP/(FP+TN)
    FNR = FN/(TP+FN)
    FDR = FP/(TP+FP)
    ACC = (TP+TN)/(TP+FP+FN+TN)
    specificity=TN/(TN+FP)
    detection_rate=TN/(TN+TP+FP+FN)
    n=len(y_test) 
    Accuracy_alex=sum(ACC)/len(ACC)*100
    recall_alex=float(sum(TPR)/len(TPR)*100)
    precision_alex=float(sum(PPV)/len(PPV)*100) 
    f1_score_alex=(2*recall_alex*precision_alex)/(precision_alex+recall_alex) 
    specificity_alex=sum(specificity)/len(specificity)*100
    mse_alex= mean_squared_error(y_test,alex_predicted)
    rootMeanSquaredErroralex = sqrt(mse_alex)
    mean_absolute_error_alex=mean_absolute_error(y_test,alex_predicted)
    print("\nAlex Net Performance : \n****************\n")
    print("Accuracy : ",Accuracy_alex)
    print("precision : ",precision_alex)
    print('Recall',recall_alex)
    print("F Measure : ",f1_score_alex)
    print("Specificity :",specificity_alex)
    print("mean squared error :",mse_alex)
    print("root Mean SquaredError  :",rootMeanSquaredErroralex)
    print("Mean Absolute Error  :",mean_absolute_error_alex)
    acc=[Accuracy_lstm,Accuracy_bilstm,Accuracy_resnet,Accuracy_cnn,Accuracy_vgg,Accuracy_alex,Accuracy_proposed]
    models=['LSTM','Bi-LSTM','ResNet','CNN','VGG','Alexnet','Proposed']
    fig = plt.figure(figsize=(8, 6))
    for i in range(len(models)):
        plt.bar(models[i],acc[i],width=0.5)    
    plt.xticks([0,1,2,3,4,5,6],models,fontname = "Times New Roman",fontweight='bold',fontsize=18)   
    plt.yticks(fontname = "Times New Roman",fontweight='bold',fontsize=18)
    plt.ylabel("Accuracy (%)",fontname = "Times New Roman",fontweight='bold',fontsize=20)
    plt.ylim(70, 100.01)
    plt.tight_layout()
    plt.show()
    plt.savefig("Results\Accuracy.png",dpi=600)
    pre=[precision_lstm,precision_bilstm,precision_resnet,precision_cnn,precision_vgg,precision_alex,precision_proposed]
    fig = plt.figure(figsize=(8, 6))
    for i in range(len(models)):
        plt.bar(models[i],pre[i],width=0.5)   
    plt.xticks([0,1,2,3,4,5,6],models,fontname = "Times New Roman",fontweight='bold',fontsize=18)   
    plt.yticks(fontname = "Times New Roman",fontweight='bold',fontsize=18)
    plt.ylabel("Precision  (%)",fontname = "Times New Roman",fontweight='bold',fontsize=20)
    plt.tight_layout()
    plt.show()
    plt.savefig("Results\Precision.png",dpi=600)
    rec=[recall_lstm,recall_bilstm,recall_resnet,recall_cnn,recall_vgg,recall_alex,recall_proposed]
    fig = plt.figure(figsize=(8, 6))
    for i in range(len(models)):
        plt.bar(models[i],rec[i],width=0.5)   
    plt.xticks([0,1,2,3,4,5,6],models,fontname = "Times New Roman",fontweight='bold',fontsize=18)   
    plt.yticks(fontname = "Times New Roman",fontweight='bold',fontsize=18)
    plt.ylabel("Recall  (%)",fontname = "Times New Roman",fontweight='bold',fontsize=20)
    plt.tight_layout()
    plt.show()
    plt.savefig("Results\Recall.png",dpi=600)
    f1_score=[f1_score_lstm,f1_score_bilstm,f1_score_resnet,f1_score_cnn,f1_score_vgg,f1_score_alex,f1_score_proposed]
    fig = plt.figure(figsize=(8, 6))
    
    for i in range(len(models)):
        plt.bar(models[i],f1_score[i],width=0.5)    
    plt.xticks([0,1,2,3,4,5,6],models,fontname = "Times New Roman",fontweight='bold',fontsize=18)   
    plt.yticks(fontname = "Times New Roman",fontweight='bold',fontsize=18)
    plt.ylabel("F-Measure  (%)",fontname = "Times New Roman",fontweight='bold',fontsize=20)
    plt.tight_layout()
    plt.show()
    plt.savefig("Results\F-Measure.png",dpi=600)
    acc=[specificity_lstm,specificity_bilstm,specificity_resnet,specificity_cnn,specificity_vgg,specificity_alex,specificity_proposed]
    models=['LSTM','Bi-LSTM','ResNet','CNN','VGG','Alexnet','Proposed']
    fig = plt.figure(figsize=(8, 6))
    for i in range(len(models)):
        plt.bar(models[i],acc[i],width=0.5)    
    plt.xticks([0,1,2,3,4,5,6],models,fontname = "Times New Roman",fontweight='bold',fontsize=18)   
    plt.yticks(fontname = "Times New Roman",fontweight='bold',fontsize=18)
    plt.ylabel("Specificity  (%)",fontname = "Times New Roman",fontweight='bold',fontsize=20)
    plt.ylim(70, 100.05)
    plt.tight_layout()
    plt.show()
    plt.savefig("Results\Specificity.png",dpi=600)
    acc=[mse_lstm,mse_bilstm,mse_resnet,mse_cnn,mse_vgg,mse_alex,mse_proposed]
    models=['LSTM','Bi-LSTM','ResNet','CNN','VGG','Alexnet','Proposed']
    fig = plt.figure(figsize=(8, 6))
    for i in range(len(models)):
        plt.bar(models[i],acc[i],width=0.5)   
    plt.xticks([0,1,2,3,4,5,6],models,fontname = "Times New Roman",fontweight='bold',fontsize=18)   
    plt.yticks(fontname = "Times New Roman",fontweight='bold',fontsize=18)
    plt.ylabel("MSE (%)",fontname = "Times New Roman",fontweight='bold',fontsize=20)
    plt.tight_layout()
    plt.show()
    plt.savefig("Results\MSE.png",dpi=600)
    acc=[rootMeanSquaredErrorlstm,rootMeanSquaredErrorbilstm,rootMeanSquaredErrorresnet,rootMeanSquaredErrorcnn,rootMeanSquaredErrorvgg,rootMeanSquaredErroralex,rootMeanSquaredErrorproposed]
    models=['LSTM','Bi-LSTM','ResNet','CNN','VGG','Alexnet','Proposed']
    fig = plt.figure(figsize=(8, 6))
    for i in range(len(models)):
        plt.bar(models[i],acc[i],width=0.5)    
    plt.xticks([0,1,2,3,4,5,6],models,fontname = "Times New Roman",fontweight='bold',fontsize=18)   
    plt.yticks(fontname = "Times New Roman",fontweight='bold',fontsize=18)
    plt.ylabel("RMSE (%)",fontname = "Times New Roman",fontweight='bold',fontsize=20)
    plt.tight_layout()
    plt.show()
    plt.savefig("Results\RMSE.png",dpi=600)
    acc=[mean_absolute_error_lstm,mean_absolute_error_bilstm,mean_absolute_error_resnet,mean_absolute_error_cnn,mean_absolute_error_vgg,mean_absolute_error_alex,mean_absolute_error_proposed]
    models=['LSTM','Bi-LSTM','ResNet','CNN','VGG','Alexnet','Proposed']
    fig = plt.figure(figsize=(8, 6))
    for i in range(len(models)):
        plt.bar(models[i],acc[i],width=0.5)   
    plt.xticks([0,1,2,3,4,5,6],models,fontname = "Times New Roman",fontweight='bold',fontsize=18)   
    plt.yticks(fontname = "Times New Roman",fontweight='bold',fontsize=18)
    plt.ylabel("MAE (%)",fontname = "Times New Roman",fontweight='bold',fontsize=20)
    plt.tight_layout()
    plt.show()
    plt.savefig("Results\mae.png",dpi=600)
    from sklearn.preprocessing import LabelBinarizer
    lb = LabelBinarizer()
    y_test=lb.fit_transform(y_test)
    pred=lb.fit_transform(pred)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(Class)):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    fig=pyplot.figure(figsize=(8,6))
    for i in range(len(Class)):
        pyplot.plot(fpr[i],tpr[i], linestyle='-',label=Class[i])
    pyplot.plot([0,1],[0,1], linestyle='-')
    pyplot.xlabel('False Positive Rate',fontname = "Times New Roman",fontsize=18,fontweight='bold')
    pyplot.ylabel('True Positive Rate',fontname = "Times New Roman",fontsize=18,fontweight='bold')
    
    pyplot.yticks(fontname = "Times New Roman",fontsize=16,fontweight='bold')
    pyplot.xticks(fontname = "Times New Roman",fontsize=16,fontweight='bold')

    legend_properties = {'weight':'bold'}
    pyplot.legend(fontsize = 14,prop=legend_properties)
    pyplot.tight_layout()

    pyplot.show()
    pyplot.savefig("Results\ROC.png",dpi=600)
    
    Train_Loss,Test_Loss,Train_Accuracy,Test_Accuracy=model_acc_loss()

    plt.figure(figsize=(8, 6))
    
    plt.plot(Train_Loss,'b', label='Train')
    plt.plot(Test_Loss, 'r', label='Test')
    
    plt.yticks(fontname = "Times New Roman",fontsize=16,fontweight='bold')
    plt.xticks(fontname = "Times New Roman",fontsize=16,fontweight='bold')
    plt.ylabel('Loss',fontsize=14,fontweight='bold')
    plt.xlabel('Epoch',fontsize=14,fontweight='bold')
    plt.legend(prop={'weight':'bold'})
    plt.tight_layout()
    plt.savefig("Results/training_loss.png",dpi=600)
    plt.figure(figsize=(8, 6))
   
    plt.plot(Train_Accuracy,'b', label='Train')
    plt.plot(Test_Accuracy, 'r', label='Test')

    
    plt.xlabel('Epoch',fontsize=14,fontweight='bold')
    plt.ylabel('Accuracy',fontsize=14,fontweight='bold')
    plt.yticks(fontname = "Times New Roman",fontsize=16,fontweight='bold')
    plt.xticks(fontname = "Times New Roman",fontsize=16,fontweight='bold')
    
    plt.legend(prop={'weight':'bold'})
    plt.tight_layout()
    plt.savefig("Results/training_acc.png",dpi=600)
    
    
    




