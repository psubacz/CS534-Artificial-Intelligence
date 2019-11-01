# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 16:58:58 2019

@author: Eastwind
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn import metrics
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import confusion_matrix

def generate_confusion_matrix(labels,hypothsis,K):
    accuracy_mtx = np.zeros((K, K))
    for i in range(0,len(labels)):
        accuracy_mtx[labels[i]][labels[i]] += 1
    con_mtx = confusion_matrix(labels, hypothsis)
    
    n_confusion_mtx = (np.identity(10))*-1
        #tally the votes for the highest number number of votes for a class.
    for c in range(0,K):
        dc = np.where(con_mtx[c] == max(con_mtx[c]))
        n_confusion_mtx[dc[0][0]] += con_mtx[c]
    print('Cluster Confusion Matrix:')
    print(n_confusion_mtx)
    print('Fowlkes-Mallows Scores')
    labels_true = accuracy_mtx.flatten()
    labels_pred = n_confusion_mtx.flatten()
    score = metrics.fowlkes_mallows_score(labels_true, labels_pred)
    print('Accuracy Score:%f'%(score*100)) 

if __name__ == '__main__':
    
    digits = load_digits()    
    #The number of clusters the data is dealing with
    centers = []
    for i in digits.target_names:   
        centers.append(digits.data[i])
    
    af = AffinityPropagation(preference=-50000,max_iter=500,).fit(digits.data)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_
    n_clusters_ = len(cluster_centers_indices)
    print('Estimated number of clusters: %d' % n_clusters_)
    generate_confusion_matrix(digits.target,af.labels_,len(digits.target_names))