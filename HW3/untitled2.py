# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 00:31:58 2019

@author: Eastwind
"""
from sklearn.model_selection import train_test_split
import numpy as np
import random as rd
import csv
import matplotlib.pyplot as plt
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

def sigmoid_function(z):
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
    M = len(Y)
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
    
def gradient_descent(W,X,Y,alpha,reg,lam):
    '''
        
    '''
    #Initialize vectors for predictions, H, and cost, C,
    H = np.zeros(len(X))
    C = np.zeros(len(X))
    G = np.zeros(len(W))
    R = 0
    #Calculate the prediction of the model
    H = model_prediction(W,X)
    H = np.reshape(H,(len(H),1))
    #Calculate the cost of the predictions
    for i in range(0,len(H)):
        C[i] = cost_function(H[i],Y[i])
    C = sum(C)/len(C)
#    print((sum(H-Y)*X).shape)
#    if(reg == True):
#        #Add the regularizer to the cost
#        C += sum(regularizer(W,lam))/(2*len(X))
#        G = np.dot(np.transpose(X),H-Y)
#        G+=(G*lam*W**2)
#        W = W-G
#    else:
    G = np.dot(np.transpose(X),H-Y)
    W = W-G
    return W,C

def logistical_regression(num_iters,W,X,Y,alpha,reg,lam):
    '''
    '''
    CH = []
    for i in num_iters:
        W,C = gradient_descent(W,X,Y,alpha,reg,lam)
        if i % 100 == 0:
            print("Iter: %d, Cost: %f" % (i,C))
            CH.append(C)
#        if i ==5:
#            break
    return CH,W

if __name__ == '__main__':
    attributes = ['age','bp','sg','al','su','rbc','pc','pcc','ba','bgr','bu',
                  'sc','sod','pot','hemo','pcv','wc','rc','htn','dm','cad',
                  'appet','pe','ane','class']
    X,Y = parse_data_file('chronic_kidney_disease_full.csv',attributes)
    #split the dataset into 80% training data and %20 testing data
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    #Initialize the weights to a (23,1)    
    W = np.ones((len(X_train[0]),1))
    max_iter = 25000
    iters = np.arange(0,max_iter)
    C,W  =logistical_regression(iters,W,X_train,y_train,0.005,True,0.2)
    # plot the cost 
    fig, ax = plt.subplots()
    ax.plot(np.arange(0,len(C)), C)
    ax.set(xlabel='Run', ylabel='Cost', title='Logistical Regression Learning Cost')
    ax.set_ylim(0, max(C)*1.1)
    plt.show()