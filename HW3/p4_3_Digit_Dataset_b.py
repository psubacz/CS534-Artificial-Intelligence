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
from sklearn.cluster import AgglomerativeClustering

def generate_confusion_matrix(labels,hypothsis,K):
    confusion_mtx = np.zeros((K, K))
    accuracy_mtx = np.zeros((K, K))
    for i in range(0,len(labels)):
        confusion_mtx[hypothsis[i]][labels[i]] += 1
        accuracy_mtx[labels[i]][labels[i]] += 1
        
    #Make new identitiy matrix to reshuffle the confusion matrix
    n_confusion_mtx = (np.identity(10))*-1
    
    #tally the votes for the highest number number of votes for a class.
    for c in range(0,K):
        dc = np.where(confusion_mtx[c] == max(confusion_mtx[c]))
        n_confusion_mtx[dc[0][0]] += confusion_mtx[c]
    print('Cluster Confusion Matrix:')
    print(n_confusion_mtx)
        #Generate the accuracy matrix to get scores
    print('Fowlkes-Mallows Scores')
    labels_true = accuracy_mtx.flatten()
    labels_pred = n_confusion_mtx.flatten()
    score =metrics.fowlkes_mallows_score(labels_true, labels_pred)
    print('Accuracy Score:%f'%(score*100)) 

if __name__ == '__main__':
    num_of_clusters = 10 #The number of clusters the data is dealing with
    threshold = 0.0000001 # the delta that needs to be exceeded to continue running. Lower is not necessarily better
    max_runs = 100 #max number of runs
    
    digits = load_digits()
    clustering = AgglomerativeClustering(n_clusters=len(digits.target_names)).fit(digits.data)
    hypothsis = clustering.labels_
    print('Number of Clusters: %d'%num_of_clusters)
    generate_confusion_matrix(digits.target,hypothsis,len(digits.target_names))