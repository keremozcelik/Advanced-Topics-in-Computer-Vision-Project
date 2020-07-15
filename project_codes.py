# -*- coding: utf-8 -*-
"""
Created on Sat May 16 13:58:52 2020

@author: Ben
"""
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt 
import os
from time import time
import seaborn as sns
import cv2
from skimage.feature import hog

#%% import dataset and Feature Extraction with HOG
train_path=["satellite-images-of-hurricance-damage/train_another/damage/","satellite-images-of-hurricance-damage/train_another/no_damage/"]
test_path=["satellite-images-of-hurricance-damage/test/damage/","satellite-images-of-hurricance-damage/test/no_damage/"]
validation_path=["satellite-images-of-hurricance-damage/validation_another/damage/","satellite-images-of-hurricance-damage/validation_another/no_damage/"]

train_features=[]
test_features=[]
validation_features=[]

for train in train_path:
    filelist = os.listdir(train)
    for file in filelist:
        gray_image = cv2.cvtColor(np.asarray(Image.open(train+file)), cv2.COLOR_BGR2GRAY)
        fd, hog_image = hog(gray_image, orientations=8, pixels_per_cell=(16, 16),cells_per_block=(4, 4), visualize=True)
        train_features.append(fd)
         
for test in test_path:
    filelist = os.listdir(test)
    for file in filelist:
        gray_image = cv2.cvtColor(np.asarray(Image.open(test+file)), cv2.COLOR_BGR2GRAY)
        fd, hog_image = hog(gray_image, orientations=8, pixels_per_cell=(16, 16),cells_per_block=(4, 4), visualize=True)
        test_features.append(fd)

#It opens when validation data is used instead of test data.      
#for validation in validation_path:
#    filelist = os.listdir(validation)
#    for file in filelist:
#        gray_image = cv2.cvtColor(np.asarray(Image.open(validation+file)), cv2.COLOR_BGR2GRAY)
#        fd, hog_image = hog(gray_image, orientations=8, pixels_per_cell=(16, 16),cells_per_block=(4, 4), visualize=True)
#        validation_features.append(fd)
        
#%%
# damage images label = 1   no damage images label = 0    
X_train = np.asarray(train_features)
z = np.zeros(5000,dtype=int)
o = np.ones(5000,dtype=int)
Y_train = np.concatenate((o, z), axis=0).reshape(X_train.shape[0],1)

X_test= np.asarray(test_features)
z2 = np.zeros(1000,dtype=int)
o2 = np.ones(1000,dtype=int)
Y_test = np.concatenate((o2, z2), axis=0).reshape(X_test.shape[0],1)

#X_valid= np.asarray(validation_features)
#z2 = np.zeros(1000,dtype=int)
#o2 = np.ones(1000,dtype=int)
#Y_valid = np.concatenate((o2, z2), axis=0).reshape(X_valid.shape[0],1)

#%% Adjust Dataset Randomly

#dataset shuffle randomly
df_train = pd.DataFrame(np.concatenate((X_train,Y_train),axis=1)).sample(frac=1).reset_index(drop=True)
df_test = pd.DataFrame(np.concatenate((X_test,Y_test),axis=1)).sample(frac=1).reset_index(drop=True)
#df_valid = pd.DataFrame(np.concatenate((X_valid,Y_valid),axis=1)).sample(frac=1).reset_index(drop=True)

#split Train,Test,Validation data like X and Y
x_train = df_train.iloc[:,:-1] #train features
y_train = df_train.iloc[:,-1] #last column of df_train = train labels
x_test = df_test.iloc[:,:-1] #test features
y_test = df_test.iloc[:,-1] #last column of df_test = test labels
#x_valid = df_valid.iloc[:,:-1] 
#y_valid = df_valid.iloc[:,-1]

#%% Naive Bayes 
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB(var_smoothing=1e-12)
start=time()
nb.fit(x_train,y_train)
nb_score = nb.score(x_test,y_test)
stop=time()
nb_time=stop-start
print("accuracy of naive bayes algo: ",nb_score)
print("time:",nb_time)
y_pred_nb = nb.predict(x_test)

#%% Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="gini",max_features=10,splitter="best",random_state=42,min_samples_split=2,min_samples_leaf=1)
start=time()
dt.fit(x_train,y_train)
dt_score = dt.score(x_test,y_test)
stop=time()
dt_time = stop-start
print("accuracy of decision tree algo: ",dt_score)
print("time:",dt_time)
y_pred_dt = dt.predict(x_test)

#%% Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators =150,max_features=10,random_state=42,criterion="entropy",min_samples_split=2,min_samples_leaf=1) 
start=time()
rf.fit(x_train,y_train)
rf_score = rf.score(x_test,y_test)
stop=time()
rf_time = stop-start
print("accuracy of random forest algo: ",rf_score)
print("time:",rf_time)
y_pred_rf = rf.predict(x_test)

#%%SVM
from sklearn.svm import SVC
svm = SVC(random_state = 42,gamma="scale",C=10,max_iter=1000)
start=time()
svm.fit(x_train,y_train)
svm_score = svm.score(x_test,y_test)
stop=time()
svm_time=stop-start
print("accuracy of svm algo: ",svm_score)
print("time:",svm_time)
y_pred_svm = svm.predict(x_test)

#%% Result table

# rows = algorithms
df = pd.DataFrame([[nb_score,nb_time],[dt_score,dt_time],[rf_score,rf_time],[svm_score,svm_time]], index=["Naive Bayes","Decision Tree","Random Forest","SVM"],
                  columns=["accuracy score","run time(second)"])

# columns = algorithms
df2 = pd.DataFrame([[nb_score*100,dt_score*100,rf_score*100,svm_score*100],[nb_time,dt_time,rf_time,svm_time]], index=["accuracy score(%)","run time(second)"],
                  columns=["Naive Bayes","Decision Tree","Random Forest","SVM"])

print(df2)

#%% confusion matrix
from sklearn.metrics import confusion_matrix
cm_nb = confusion_matrix(y_test,y_pred_nb)
cm_dt = confusion_matrix(y_test,y_pred_dt)
cm_rf = confusion_matrix(y_test,y_pred_rf)
cm_svm = confusion_matrix(y_test,y_pred_svm)

list1=[[cm_nb,"Naive Bayes"],[cm_dt,"Decision Tree"],[cm_rf,"Random Forest"],[cm_svm,"SVM"]]

for i,k in list1:
    f, ax = plt.subplots(figsize = (5,5))
    sns.heatmap(i,annot = True, linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax) 
    plt.xlabel("Prediction")
    plt.ylabel("Test data")
    plt.title("{} Confusion Matrix".format(k))
    plt.show()

#%% Roc curve analysis
from sklearn import metrics

fpr_nb,tpr_nb,thresholds_nb= metrics.roc_curve(y_test,y_pred_nb)
fpr_rf,tpr_rf,thresholds_rf= metrics.roc_curve(y_test,y_pred_rf)
fpr_dt,tpr_dt,thresholds_dt= metrics.roc_curve(y_test,y_pred_dt)
fpr_svm,tpr_svm,thresholds_svm= metrics.roc_curve(y_test,y_pred_svm)

plt.figure()
plt.plot(fpr_nb, tpr_nb , color="red" , label="Naive Bayes ROC curve under area = %0.3f" % nb_score)
plt.plot(fpr_rf, tpr_rf , color="blue", label="Random Forrest ROC curve under area =  %0.3f" % rf_score)
plt.plot(fpr_dt, tpr_dt , color="green", label="Decision Tree ROC curve under area = %0.3f" % dt_score)
plt.plot(fpr_svm, tpr_svm , color="yellow", label="SVM ROC curve under area = %0.3f" % svm_score)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Analysis')
plt.legend()

#%% Grid Search Part
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, StratifiedKFold

random_state = 42
classifier = [GaussianNB(),DecisionTreeClassifier(random_state=random_state),
             RandomForestClassifier(random_state=random_state),SVC(random_state=random_state)
             ]

nb_param_grid = {"var_smoothing":[1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7]}

dt_param_grid = {"max_features":[10],
                "min_samples_split" :[2,3,10],
                "min_samples_leaf":[1,3,10],
                "criterion":["gini","entropy"],
                "splitter":["best"]}

rf_param_grid = {"max_features":[10],
                "min_samples_split":[2,3,10],
                "min_samples_leaf":[1,3,10],
                "n_estimators":[100,120,150],
                "criterion":["gini","entropy"]}

svc_param_grid = {"kernel":["rbf"],
                 "gamma":["scale","auto"],
                 "max_iter":[1000],
                 "C":[1,3,10]}

classifier_param=[nb_param_grid,dt_param_grid,rf_param_grid,svc_param_grid]

cv_result = []
best_estimators = []
best_params=[]
for i in range(len(classifier)):
    clf = GridSearchCV(classifier[i],param_grid=classifier_param[i],cv=StratifiedKFold(n_splits=3),scoring="accuracy",n_jobs=-1,verbose = 1)
    clf.fit(x_train,y_train)
    cv_result.append(clf.best_score_)
    best_estimators.append(clf.best_estimator_)
    best_params.append(clf.best_params_)
    print(cv_result[i])