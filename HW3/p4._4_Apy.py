# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 22:43:41 2019

@author: Eastwind
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#a. Support Vector Machine with the linear kernel and default parameters (sklearn.svm.SVC).

def parse_data_file(data_file,attributes):
    '''
    Parses the data file as a tab seperated value file
    Returns an array of x-y values
    '''    
    #Create nested lists for each attribute
    data = []
    X = []
    Y = []
    #Open the data file and generate a list of data from teh tsv
    with open(data_file) as tsv:
        for line in csv.reader(tsv, delimiter=","):
            #Encode the data to ensure numerical methods. Unknown data is kept
            line = encode_data(line)
            data.append(line)
    
    #The data is not time dependant and can be shuffled. 
    rd.shuffle(data)
    for line in data:
        X.append(line[0:len(line)-2])
        Y.append([line[-1]])
    return np.array(X),np.array(Y)

def encode_data(data):
    '''
    Encode data to be all numerical. Boolean esque values are set to 0 or 1 
     depending on code. 
    '''
    for i in range(0,len(data)):
        if (data[i]== 'normal'):
            data[i] = 0
        elif (data[i]== 'abnormal'):
            data[i] = 1
        elif (data[i]== 'notpresent'):
            data[i] = 0
        elif (data[i]== 'present'):
            data[i] = 1
        elif (data[i]== 'no'):
            data[i] = 0
        elif (data[i]== 'yes'):
            data[i] = 1
        elif (data[i]== 'good'):
            data[i] = 0
        elif (data[i]== 'poor'):
            data[i] = 1
        elif (data[i]== 'notckd'):
            data[i] = 0
        elif (data[i]== 'ckd'):
            data[i] = 1
        elif (data[i]== '?'):
            data[i] = 0
        data[i] = float(data[i])
    return data

def f_measure(H,Y):
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    for i in range(0,len(H)):
        if(Y[i]==1 and H[i]==1):
            TP+=1 
        if(Y[i]==0 and H[i]==0):
            TN+=1 
        if(Y[i]==0 and H[i]==1):
            FN+=1
        if(Y[i]==1 and H[i]==0):
            FP+=1
    
    pre = TP/(TP+FP)
    rec = TP/(TP+FN)
    return (2*pre*rec)/(pre+rec)

attributes = ['age','bp','sg','al','su','rbc','pc','pcc','ba','bgr','bu',
                  'sc','sod','pot','hemo','pcv','wc','rc','htn','dm','cad',
                  'appet','pe','ane','class']

X,Y = parse_data_file('chronic_kidney_disease_full.csv',attributes)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
classifer = SVC(gamma='auto')
classifer.fit(X_train, y_train)
hypo = classifer.predict(X_train)
score0 = f_measure(hypo, y_train)
hypo = classifer.predict(X_test)
score1 = f_measure(hypo,y_test)
print('Train F-Measure: %f' % score0)
print('Test F-Measure: %f' % score1)