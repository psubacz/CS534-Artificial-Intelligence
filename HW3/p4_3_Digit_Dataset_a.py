# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 16:58:58 2019

@author: Eastwind
"""

import numpy as np
import random as rd
import matplotlib.pyplot as plt
import csv, time
from sklearn.datasets import load_digits
from sklearn import metrics

class digit:
    def __init__(self,data,label):
        self.data = data 
        self.label = label
        
def euclidean_distance(d1,d2):
    '''
    Returns the euclidean distance
    '''
#    print(d1)
    n1 = d1[0].data
    n2 = d2
    n3 = n1-n2
    n4 = 0
    for i in range(0,len(n3)):
        n4 += n3[i]**2
    return np.sqrt(n4)

def generate_cluster_labels(classes,K):
    #def generate_cluster_labels(classes,K):
    #Each cluster is defined by the majority represented cluster
    class_labels = []
    confusion_mtx = []
    i = 0
    for c in classes:
        class_labels.append([])    
        confusion_mtx.append(np.zeros(K))
        confusion_mtx[i][i] = -1
        i+=1
        
    #tally the votes for the highest number number of votes for a class.
    for c in classes:
        votes =  np.zeros(K)
        for i in range(0,len(c)):
            votes[c[i].label] +=1
         
        dc = np.where(votes ==max(votes))
        confusion_mtx[dc[0][0]] += votes
        print('Cluster: %d has %d votes. This cluster has been labeled: C-%d ' % (i,votes[dc[0]],dc[0]))
    print('Cluster Confusion Matrix:')
    confusion_mtx = np.array(confusion_mtx)
    print(confusion_mtx)
    
    #Generate the accuracy matrix to get scores
    print('Fowlkes-Mallows Scores')
    labels_true = (np.identity(10)*confusion_mtx).flatten()
    labels_pred = confusion_mtx.flatten()
    score =metrics.fowlkes_mallows_score(labels_true, labels_pred)
    print('Accuracy Score:%f'%(score*100)) 
            
def k_means_clustering(k,data,threshold,pointbreak):
    #Number of execution cycles
    t = 0 
    #Data to be worked with of length 64
    X = data.data
    #Labels to evaluate the data from 0 -> 9
    Y = data.target
    #Generate number of clusters
    m = np.arange(k)
    #Create clusters and initialize to localize to zero
    theta_j = []
    #Generate empty starting cluster
    for j in m:
        theta_j.append([])

    #Seed the initial clusters with 1 datum
    for i in range(0,len(theta_j)):
        theta_j[i].append(digit(X[i],Y[i]))
    
    d_theta = threshold*2
    #run until cluter date is less than threshold meaning that the clusters no longer move
    while(d_theta > threshold):
        #Create a empty list store distance values
        u = [None] * len(theta_j)
        classes = []
        for i in range(0,k):
            classes.append([])
        #Increment the execution counter
        t +=1 
        #If the max number of runs has exceeded the set value, return failure
        if (t>=pointbreak):
            print('Exceeded number of runs...')
            break
        #loop through the data and Calculate the distance to each centroid
        for i in range(0,len(X)):    
            for ii in range(0,len(theta_j)):
                #Calculate the l2 distance of each pixel and return the disanace to each cluster
                u[ii] = euclidean_distance(theta_j[ii],X[i])
            #Find the closest distance
            z= u.index(min(u))
            #Assign data point to closest cluster
            classes[z].append(digit(X[i],Y[i]))                    
        _theta = []
        #Define initial threshold
        
        for i in range(0,len(classes)):
            #Calculate the new center of each cluster for each cluster
            d = np.zeros(len(X[0]))
            #Calculate the new pixel distance by averageing each pixel element
            for ii in range(0,len(classes[i])):
                d +=  classes[i][ii].data
            d /= len(classes[i])
            _theta.append(np.sqrt((sum(d)-sum(theta_j[i][0].data))**2))

            #Update new cluster center
            theta_j[i] = [digit(d,i)]
        n_theta =float(max(_theta))
        
        if (n_theta > d_theta):
            d_theta = n_theta           
        r = 'K-Means Run '+str(t)
        print(r)
        
        if (n_theta < threshold):
            print('Jobs done!')
            
            return classes,theta_j

if __name__ == '__main__':
    num_of_clusters = 10 #The number of clusters the data is dealing with
    threshold = 0.1 # the delta that needs to be exceeded to continue running. Lower is not necessarily better
    max_runs = 100 #max number of runs
    
    digits = load_digits()    
    classes,theta_j = k_means_clustering(num_of_clusters,digits,threshold,100)
    print('Number of Clusters: %d'%num_of_clusters)
    #Generate the labels for each cluster
    labels = generate_cluster_labels(classes,num_of_clusters)
#            confusion_matrix = generate_confusion_matrix(classes)
#            accuracy = calculate_accuracy()