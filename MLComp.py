# -*- coding: utf-8 -*-
"""
Created on Wed Oct 05 08:35:58 2016

@author: rwanjohi
"""

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn.ensemble  import GradientBoostingClassifier as GBC, RandomForestClassifier as RF, AdaBoostClassifier as ADA
from sklearn.naive_bayes import GaussianNB as NB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import BernoulliRBM as nnet
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier as SGDC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import cross_val_score
 
from sklearn.svm import SVC
from sklearn import svm, tree
from sklearn import metrics, datasets
from sklearn.datasets import make_moons, make_circles, make_classification
import pandas as pd
import numpy as np
import scipy, math
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV
import time
from xgboost import XGBClassifier

#-------------------------------------------------------------------------------------------------------------------------

﻿def MLfun(df, Models, names, prop = 0.2):
    X = pd.get_dummies(df[df.columns[:-1]])
    Y = pd.get_dummies(df[df.columns[-1]]).as_matrix()[:,1]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = prop)
    print( "*"*100 )  
    print( "   if you please: Your Name \nYour title \nYour company " )
    print( "*"*100 )
    plt.figure(figsize = (15, 10))              # you can adjust this 
    # iterate over classifiers
    for name, model in zip(names, Models):
        start = time.time()
        model.fit(X_train, Y_train)
        score = model.score(X_train, Y_train)
        Y_pred = model.predict(X_test)
        Acc = metrics.accuracy_score(Y_test, Y_pred)
        probs   = model.predict_proba(X_test)
        preds = probs[:,1]
        fpr, tpr, threshold = metrics.roc_curve(Y_test, preds)
        roc_auc = metrics.auc(fpr, tpr)
        Acc = metrics.accuracy_score(Y_test, Y_pred)
        F1  = metrics.f1_score(Y_test, Y_pred)
        precision = metrics.precision_score(Y_test, Y_pred)
        Ypred = model.predict_log_proba(X_test)
        auc = roc_auc_score(Y_test, Ypred[:,1] - Ypred[:,0])
        
       
        results = dict()
        results['Accuracy'] = round((Acc * 100), 4)
        results['Auc']  = round((auc*100), 4)
        results['F1']   = round((F1 * 100), 4)
        results['Precision'] = round((precision * 100), 4)
        print('\n')
        print( " Model: %s" % model)
        print( "="*100 )  
        print( "Confusion Matrix")
        print ("-"*100 )
        print ('\n')
        print( pd.crosstab(Y_test, Y_pred, rownames=['Actual'], colnames=['Predicted']))
        print( "="*100 )
        print(metrics.classification_report(Y_test, Y_pred)  ) 
        print ('\n')
        print( "="*100 )  
        print( "Other Stats:")
        print ("-"*100 )
        print ('\n')
        # print ("="*100 )
        result = pd.DataFrame([results])
        print(result)      
        print ("-"*100 )
        print ('\n')
        stop = time.time()
        duration = stop -start
        print( "Time taken: %.4f" % duration)
        print ("-"*100 )
        print( "="*100 )
        print('\n\n\n')
        plt.plot(fpr, tpr, label='%s (auc = %0.2f)' % (name, roc_auc))
       
    plt.plot([0, 1], [0, 1] , 'k--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for Classifiers:')
    plt.legend(loc="lower right")
    plt.show()   

 
#---------------------------------------------------------------------------------------------------------------------------

Models = [
    LogisticRegression(C=1, penalty = 'l1'),                          #  L1 norm (Lasso)
    LogisticRegression(C=1, penalty = 'l2'),                          #  L2 norm (Ridge); default in sklearn.linear_model
    DecisionTreeClassifier(),                                         #  Decision Tree
    GBC(n_estimators= 100),                                           #  Gradient Boosting                                                 
    NB(),                                                             #  Naive Bayes (Gaussian)                                                
    ADA(n_estimators= 500),                                           # Adaptive Boosting (Decison Tree)                                            
    RF(n_estimators = 500)]                                           #  Random Forest
    #SVC(kernel= ...., C=...., gamma = .., probability = True),       #  Support vector machine
    #XGBClassifier(),                                                 # Extreme boosting 
    #nnet(......),                                                    #  Neural Network
    # KNeighborsClassifier(3),                                        #  Nearest Neighbors"
    # QuadraticDiscriminantAnalysis(),                                # "QDA"        
    # etc ]                               

﻿#-----------------------------------------------------------------------------------
﻿# name your models,
names = [ "L1 Logistic",
          "L2 Logistic",
          "Decision Tree",
          "Gradient Boosting",
          "Naive Bayes",
          "AdaBoost (Decision Tree)",
          "Random Forest",
          ] 

﻿#--------------------------------------------------------------------------------------------------------------------------
﻿# prop= proportion of Test, Train datasets. My default = 20:80﻿

# Read the data
 
df =  pd.read_csv('path/to/your/dataset.csv')

 
## Run 
﻿
%time  MLfun(df, Models, names, prop )

﻿﻿
