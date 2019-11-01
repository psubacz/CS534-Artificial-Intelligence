# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 02:28:43 2019

@author: Eastwind
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 00:31:58 2019

@author: Eastwind

https://www.ritchieng.com/logistic-regression/#2c-advanced-optimization
https://thelaziestprogrammer.com/sharrington/math-of-machine-learning/the-gradient-a-visual-descent
https://thelaziestprogrammer.com/sharrington/math-of-machine-learning/solving-logreg-newtons-method
http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=MachineLearning&doc=exercises/ex2/ex2.html
https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c

"""
from sklearn.model_selection import train_test_split
import numpy as np
import random as rd
import csv
import matplotlib.pyplot as plt


def parse_data_file(labels,data_file,attributes,standardize):
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
            if line is not None:
                data.append(line)

    #The data is not time dependant and can be shuffled.
    for line in data:
        X.append(line[0:len(line)-1])
        Y.append([line[-1]])
    X = np.array(X)
    Y = np.array(Y)
    #Calculate missing valeus and subsitute into missing field
#    X = plug_missing_values(X.T,labels).T
    if standardize:
        X = standardization(X.T,labels).T
    return X, Y

def plug_missing_values(dataset,labels):
    '''
    plugs missing data by calculating the mean feature for each feature set.
    '''

    for i in range(0,dataset.shape[0]):
        counter = 0
        mean = 0
        for ii in range(0,dataset.shape[1]):
            if dataset[i][ii] >=0:
#                print(dataset[i][ii])
                counter +=1
                mean+=dataset[i][ii]
        mean/=counter
        
        for ii in range(0,dataset.shape[1]):
#            print(dataset[i][ii])
            if dataset[i][ii] <0:
                
                dataset[i][ii] = mean
        for ii in range(0,dataset.shape[1]):
            if labels[i]==False:
                if mean>=0.5:
                    dataset[i][ii] = 1
                else:
                    dataset[i][ii] = 0
            dataset[i][ii] = float(dataset[i][ii])
    return dataset

def standardization(dataset,labels):
    for i in range(0,dataset.shape[0]):
        if labels[i] == True:
            min_ = min(dataset[i])
            max_ = max(dataset[i])
            for ii in range(0,dataset.shape[1]):
                dataset[i][ii] = (dataset[i][ii]-min_)/(max_-min_)
    return dataset

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

def sigmoid_function(z):
#    print(z)
#    for i in range(0,320):
#        print(i, z[i])
#        print((1/(1+np.exp(-z[i]))))
    return (1/(1+np.exp(-z)))

def model_prediction(W,X):
    '''
    X is the data being evaluated of size (N,M)
    W is the weights of the function of size (M,1)
    where N is the number of samples and M in the number of data points

    Returns the scalar sigmoid output
    '''
    return sigmoid_function(np.dot(X,W))

def cost_function(H,Y):
    '''
    Computes the cost function of the of training values. Needs to be implemented as a piecewise
    function to avoid numpy nan and inf value

    '''
#    For y=0:
#        ─  if hw(x)=0, cost = 0
#        ─  with hw(x) approaching 1 , cost will approach ∞
#    For y=1:
#        ─  if hw(x)=1, cost = 0
#        ─  with hw(x) approaching 0 , cost will approach ∞
#        ─  That means penalizing learning algorithm by a huge number
    
    if (Y[0]==0):
        if H == 1:
            #Avoid ∞ by setting to 1
            C = 1
        else:
            C = -1*np.log(1-H)
    elif(Y[0] == 1):
        if H == 0:
            #Avoid ∞ by setting to 1
            C = 1
        else:
            C = -1*np.log(H)
    return C

def regularizer(W,lam):
    return lam*(W**2)

def gradient_descent(W,X,Y,alpha,lam):
    '''

    '''
    #Initialize vectors for predictions, H, and cost, C,
    H = np.zeros(len(X))
    C = np.zeros(len(X))
    G = np.zeros(len(W))
    #Calculate the prediction of the model
    H = model_prediction(W,X)
    H = np.reshape(H,(len(H),1))
    #Calculate the cost of the predictions
    for i in range(0,len(H)):
        C[i] = cost_function(H[i],Y[i])
    C = (sum(C)/len(C))+sum(regularizer(W,lam))/(2*len(X))

    d_C = (np.dot(np.transpose(X),H-Y)+(lam*W))/len(X)

    Hess = (np.dot((X.T),H*(np.ones((len(H),1)))*X))+(lam*W)/len(X)*np.ones((len(W),len(W)))
    G = (Hess**-1).dot(d_C)
    W = W-G
    return H,W,C

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
#        print(TP,TN,FN,FP)
    if TP == 0:
        return 0.00
    else:
        pre = TP/(TP+FP)
        rec = TP/(TP+FN)
        return (2*pre*rec)/(pre+rec)

def _predict(W,X,Y):
    H = model_prediction(W,X)
    perf = f_measure(H,Y)

    return perf

def logistical_regression(num_iters,W,X,Y,alpha,lam_r):
    '''
    '''
    hist = []
    for i in num_iters:
        H,W,C = gradient_descent(W,X,Y,alpha,lam_r)
        perf = f_measure(H,Y)
#        break
        if i % 100 == 0:
            hist.append(perf)
    return hist,W

if __name__ == '__main__':
    attributes = ['age','bp','sg','al','su','rbc','pc','pcc','ba','bgr','bu',
                  'sc','sod','pot','hemo','pcv','wc','rc','htn','dm','cad',
                  'appet','pe','ane','class']

    #Labels for numberical, vs none numberical data
    labels = [True,True,True,True,True,False,False,False,
              False,True,True,True,True,True,True,True,
              True,True,False,False,False,False,False,False,False]
    #Standardize numerical data?
    standardize = True
    X,Y = parse_data_file(labels,'chronic_kidney_disease_full.csv',attributes,standardize)
    #split the dataset into 80% training data and %20 testing data
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    max_iter = 1000
    iters = np.arange(0,max_iter)
    lamb = -2

    lr_preformance_test = []
    for i in range(0,31):
        lr_preformance_test.append([])
        #Initialize the weights to a (23,1)
        W = np.ones((len(X_train[0]),1))*2
        _,W  =logistical_regression(iters,W,X_train,y_train,0.005,lamb)
        P_test = _predict(W,X_test,y_test)
        lr_preformance_test[i].append(P_test)
        print("Lamda: %f, Preformance: %f" % (lamb,P_test*100))
        lamb +=0.2

#    # plot the cost
    fig, ax = plt.subplots()
    ax.plot(np.arange(0,len(lr_preformance_test)), lr_preformance_test)
    ax.set(xlabel='Run', ylabel='Performance', title='Logistical Regression Preformance')
    plt.show()