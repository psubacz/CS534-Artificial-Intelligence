# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 18:45:45 2019

@author: psubacz

Homework 4, Problem 1 - K-Mean Algorithm

This scrpt implements a K-Mean Clustering Algorithm on a given set of XY data.
"""

import numpy as np
import random as rd
import matplotlib.pyplot as plt
import csv

def euclidean_distance(x1,y1,x2,y2):
    '''
    Returns the euclidean distance
    '''
    return np.sqrt((x1-x2)**2+(y1-y2)**2)
    
def parse_data_file(data_file):
    '''
    Parses the data file as a tab seperated value file
    Returns an array of x-y values
    '''
    x = []
    y = []
    data = []
    #Open the data file and generate a list of data from teh tsv
    with open(data_file) as tsv:
        for line in csv.reader(tsv, delimiter="\t"):
            x.append(float(line[1]))
            y.append(float(line[2]))
            data.append((x[-1],y[-1]))
    return x,y,data

def plot_data(title,x,y,clusters):
    '''
    Plots the data on a 2d graph 
    '''
    col = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    plt.scatter(x, y,s=1)
    for i in range(0,len(clusters)):
        lab = 'Cluster_'+str(i)
        plt.scatter(clusters[i][0],clusters[i][1], s=300, marker='1',label=lab, c = col[i])
    plt.title(title)
    plt.xlabel('Length')
    plt.ylabel('Width')
    plt.show()

def plot_cluster_data(title,points,clusters):
    '''
    Plots the data on a 2d graph relvent to each cluster.
    '''
    col = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    #For each Cluster
    for i in range(0,len(points)):
        x = []
        y = []
        #For each XY data point in each cluster, build a list of XY pairs
        for ii in range(0,len(points[i])):
            x.append(points[i][ii][0][0])
            y.append(points[i][ii][0][1])           
        #Plot XY data on scatter plot
        plt.scatter(x, y,s=1,c = col[i])
        lab = 'Cluster_'+str(i)
        #Plot cluster labels
        plt.scatter(clusters[i][0],clusters[i][1], s=300, marker='1',label=lab, c = col[i])
    plt.title(title)
    plt.xlabel('Length')
    plt.ylabel('Width')
    plt.legend(loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    plt.show()


def k_means_clustering(num_clusters,x_data,y_data,data,threshold,pointbreak):
    t= 0 #Number of execution cycles
    #Find extremes of each dataset
    x_max = max(x_data)
    x_min = min(x_data)
    y_max = max(y_data)
    y_min = min(y_data)
    #Generate number of clusters
    m = np.arange(num_clusters)
    #Create clusters and initialize to localize to zero
    theta_j = []
    #Ramdomly generate the starting XY values for each cluster
    for j in m:
        x_r = rd.uniform(x_min,x_max)
        y_r = rd.uniform(y_min,y_max)
        theta_j.append([x_r,y_r])
    
    #Plot initial dataset
    plot_data('Initial Graph: Random Centroid Locations',x,y,theta_j)
    
    #Define initial threshold
    d_theta = threshold*2
    #run until cluter date is less than threshold meaning that the clusters no longer move
    while(d_theta > threshold):
        #Create a empty list store distance values
        u = [None] * len(theta_j)
        w = []
        for i in range(0,num_clusters):
            w.append([])
        #Increment the execution counter
        t +=1 
        #If the max number of runs has exceeded the set value, return failure
        if (t>=pointbreak):
            print('Exceeded number of runs...')
            break
        #loop through the data and Calculate the distance to each centroid
        old_theta = [0,0]
        for i in range(0,len(data)):    
            for ii in range(0,len(theta_j)):
                u[ii] = euclidean_distance(theta_j[ii][0],theta_j[ii][1],data[i][0],data[i][1])
            #Assign data point to closest cluster
            z= u.index(min(u))
            w[z].append([data[i],min(u)])
        
        #Calculate the new center of each cluster
        d = []
        #for each cluster get the x and y data
        for i in range(0,len(w)):
            x_r = 0
            y_r = 0
            for ii in range(0,len(w[i])):
                x_r += w[i][ii][0][0]
                y_r += w[i][ii][0][1]
            #Store old theta values
            old_theta[0] = theta_j[i][0]
            old_theta[1] = theta_j[i][1]
            #calculate new theta values
            if (len(w[i]) !=0):
                theta_j[i][0] = x_r/len(w[i])
                theta_j[i][1] =y_r/len(w[i])
            #store the max movement of each cluster so that the clusters can 
            # still be updatd if a cluster is still moving
            d.append(np.abs(theta_j[i][0]-old_theta[0]))
            d.append(np.abs(theta_j[i][1]-old_theta[1]))
        d_theta = max(d) 
        
        r = 'K Means Graph:Run '+str(t)
        plot_cluster_data(r,w,theta_j)
        if not (d_theta > threshold):
            print('Jobs done!')

#Parse the data as a tsv file, return a list of tuples.
if __name__ == '__main__':
    x,y,data = parse_data_file('cluster_data.txt')
    num_of_clusters = 6 #The number of clusters the data is dealing with
    threshold = 0.0000001 # the delta that needs to be exceeded to continue running. Lower is not necessarily better
    max_runs = 100 #max number of runs
    k_means_clustering(num_of_clusters,x,y,data,threshold,100)