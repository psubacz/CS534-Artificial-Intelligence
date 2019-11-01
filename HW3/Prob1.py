# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 19:57:01 2019

@author: Eastwind
"""
import matplotlib.pyplot as plt
import numpy as np
import random as rd
import csv

class unknown_data:
    def __init__ (self,):
        self.data = []
        self.solution = []
        self.size = 0
    def add_data(self,new_data,len_data):
        self.data.append([float(i) for i in new_data])
        self.size = len_data
        
def parse_data_file(data_file):
    '''
    Parses the data file as a tab seperated value file
    Returns an array of x-y values
    '''
    data = unknown_data()
    
    #Open the data file and generate a list of data from teh tsv
    with open(data_file) as tsv:
        for line in csv.reader(tsv, delimiter="\t"):
            data.add_data(tuple(line[1:]),line[0])
    data.data = np.array(data.data).T
    return data

def euclidean_distance(A,B):
    '''
    Vectorized calculation of L2 norm. Returns the euclidean distance
    '''
    return np.sqrt(np.sum((A - B)**2))

def find_closest_cluster(data,clusters,data_size):
    '''
    
    Calculates the cluster distances
    
    '''
    clster_cnt = np.zeros((clusters.shape[0],1))
    assgn_clust =  np.zeros((data_size[1],1))
    c_dist = np.zeros((clusters.shape[0],data_size[1]))
    new_clster_ = np.zeros((clusters.shape[0],data_size[0]))
    
    for i in range(0,clusters.shape[0]):
        for ii in range(0,data_size[1]):
            c_dist[i][ii] = euclidean_distance(data.T[ii],clusters[i])
    
    #for the distance of each point
    for i in range(0,data_size[1]):
        #Get the index of the min distance calucalated
        ind = np.where(c_dist.T[i] ==  min(c_dist.T[i]))
        #and assign it to the cluster
        assgn_clust[i] = np.where(c_dist.T[i] ==  min(c_dist.T[i]))
        #Calculate new cluster centroids based on cluster assignments
        new_clster_[ind] +=data.T[i]
        clster_cnt[ind]+=1
    #Take the average of each cluster
    for i in range(0,clusters.shape[0]):
#        print(clusters[i] )
#        print(clster_cnt[i])
#        print(new_clster_[i])
        if new_clster_[i].any(0):
            new_clster_[i]= new_clster_[i]/clster_cnt[i]
        else:
            new_clster_[i] =clusters[i] 
    #Remove nans as zeros
    new_clster_ = np.nan_to_num(new_clster_)
    
#    print(new_clster_)
    return new_clster_,assgn_clust
    

def plot_data(title,data,clusters,assngmnt):
    '''
    Plots the data on a 2d graph 
    '''
    col = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf','#1f77b4']
#    print(clusters[0][0])
#    print(clusters.shape)
    
    plt.scatter(data[0], data[1],s=1)
    data = data.T   
    
    for i in range(0,len(clusters)):
        lab = 'Cluster_%d - Centroid Location: (%0.3f, %0.3f)'% (i+1,clusters[i][0],clusters[i][1])
#        lab = 'Cluster_'+str(i)+' - Centroid Location: ('+str()+','+str(clusters[i][1])+')'
        plt.scatter(clusters[i][0],clusters[i][1], s=250, marker='x',label=lab, c = col[i])
        for ii in range(0,data.shape[0]):
            if assngmnt[ii][0] == i:
                plt.scatter(data[ii][0],data[ii][1],s=1,c = col[i])
    plt.title(title)
    plt.xlabel('Length')
    plt.ylabel('Width')
    plt.legend(loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    plt.show()
    
def k_means_clustering(num_clusters,data,threshold,pointbreak,seed_clusters = None):
    data_size = data.shape
    #Generate number of clusters
    if seed_clusters ==None:
        theta_j = np.zeros((num_clusters,data_size[0]))
        #Create clusters and Ramdomly generate initial locations of clusters
        for i in range(0,num_clusters):
            for ii in range(0,data.shape[0]):
                _min = min(data[ii])
                _max = max(data[ii])
                theta_j[i][ii] = rd.uniform(_min,_max)
    else:
        theta_j = seed_clusters
    
    #Define initial threshold
    d_theta = threshold*2
    #Iteration counter
    iters = 0
    assgn_clust =  np.ones((data_size[1],1))*-1
    
    #run until cluter date is less than threshold meaning that the clusters no longer move
    while(d_theta > threshold):
        #Increment the execution counter
        iters +=1
        #If the max number of runs has exceeded the set value, return failure
        if (iters>=pointbreak):
            print('Exceeded number of runs...')
            return theta_j,assgn_clust
        else:
            r = 'K Means Graph:Run '+str(iters)
            #Calculate the L2 distance to each cluster. Returns assign data to closest clusters and new 
            #cluster centroids
            theta_j_,assgn_clust = find_closest_cluster(data,theta_j,data_size)    
            d_theta_j = theta_j - theta_j_
            d_theta = max(sum(d_theta_j.T))
            if data_size[0]<=2:
                plot_data(r,data,theta_j_,assgn_clust)
                theta_j=theta_j_
            if not (d_theta > threshold):
                print('Jobs done!')
                return theta_j,assgn_clust

if __name__ == '__main__':
    #The number of clusters the data is dealing with
    num_of_clusters = 2
    #The delta that needs to be exceeded to continue running. Lower is not necessarily better
    threshold = 0.0000001
    #max number of runs
    max_runs = 100
    #Load data
    data = parse_data_file('cluster_data.txt')
    #Run Clustering algorithm 
    theta_j,assgn_clust = k_means_clustering(num_of_clusters,data.data,threshold,100)
#    #Final data plot
#    r = 'K-Means Plot'
#    plot_data(r,data.data,theta_j,assgn_clust)
