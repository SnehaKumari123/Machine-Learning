#imported libraries
#import copy
import numpy as np
import matplotlib.pyplot as plt
from numpy import random
import pandas as pd
from mat4py import loadmat #pip install mat4py
from random import sample
from collections import defaultdict
import matplotlib.colors as pltc
import math

"""Below Api  loads the data from given input file 
and  process the data"""

def get_data():
    dataset = loadmat('project2.mat') # this is the given inout data file which has been renamed to Project2.mat
    (k, v), = dataset.items()
    X=np.array([v])
    X.resize(300,2)
    d = defaultdict(list)
    for i in v:
        d['x'].append(i[0])
        d['y'].append(i[1]) 

    df = pd.DataFrame([ v  for k, v in d.items()], )
    df=df.transpose()
    df.columns = ['x', 'y']
    all_colors = [k for k,v in pltc.cnames.items()]
    colors = sample(all_colors, 100)
    return df,colors
 
""" getting centoroids for strategy 1 which means selecting random centroids from the given samples.
Used random function to fetch random data from the given samples itself """

def get_centoroids(df,NumOfCentorids):
    c=NumOfCentorids
    centroids={}
    for i in range(1,c+1):
        r=random.randint(1,len(df)) 
        centroids[i] = [df['x'][r],df['y'][r]]  # choosing random samples
    return centroids


"""getting centoroids for strategy 2- which is pick the first center ranodomly from unform space , 
for rest centers choose a sample from the given sameple where average distance is maximum to all previous choosen centers"""
def get_centoroids_2(df,NumOfCentorids):
    centroid_list = []
    centroids=dict([])
    t=1
    centroid_list.append([np.random.uniform(0, 10), np.random.uniform(0, 10)]) #picking random center as first centroid 
    while len(centroid_list) < NumOfCentorids:
        max_distance = 0
        ideal_point = []
        for index, row in df.iterrows():
            i= [row['x'],row['y']]
            if(i in centroid_list):
                continue
            total_distance = 0
            for j in centroid_list:
                distance = calculate_distance(i,j)
                total_distance = total_distance+distance  
            # average distance of each from al other centroids
            avg_distance = total_distance/len(centroid_list) 
            
            # if average is maximum,then max_distance will be the latest maximum distance
            if avg_distance > max_distance :  
                max_distance = avg_distance
                ideal_point = i
        
        centroid_list.append(ideal_point)
    
    for i in centroid_list: 
        centroids[t]=i
        t=t+1   
    return centroids


    
"""below api -assigning data set points with centroids
calculated average distance from each point to centroid 
and based on the values, assigned each point to a  cluster centroid which is closest"""

def assignment(df, centroids):
    for i in centroids.keys():
        df['distance_from_{}'.format(i)] = (
            np.sqrt(
                (df['x'] - centroids[i][0]) ** 2
                + (df['y'] - centroids[i][1]) ** 2
            )
        )
    centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]
    
    #assigning to closest one
    df['closest'] = df.loc[:, centroid_distance_cols].idxmin(axis=1) 
    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
    
    df['color'] = df['closest'].map(lambda x: colors[x]) 
    return df

"""Plotting assignment results:
    below api plots all the given input sample """
def plot_assignment(df,centroids):    
    plt.figure(figsize=(10, 10))
    plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')
    for i in centroids.keys():
        plt.scatter(*centroids[i], color=colors[i],marker='*',s=150,linewidths=5) 
    plt.title('With {} Cluster(s)'.format(i))
    plt.xlim(-2, 10)
    plt.ylim(-2, 10)
    plt.show()

"""updating centroids by calculating mean distance
 from each point and corresponding centroid.Updating centroid with new position :
    """    
def update(centroids):
    for i in centroids.keys():
        centroids[i][0] = np.mean(df[df['closest'] == i]['x'])
        centroids[i][1] = np.mean(df[df['closest'] == i]['y'])
    return centroids

"""calculating square root distance of 2D points"""
def calculate_distance(x,y):
    return math.sqrt((pow(float(x[0])-float(y[0]),2))+(pow(float(x[1])-float(y[1]),2)))

"""getting optimazed distance from final centroid-wcss
distance between  each point and closest cluster will give the final optimaized value
which is also called as wcss-within-cluster sums of squares (WCSS)"""
def get_optimzed_values(df,centroids):
    t=0
    for index, row in df.iterrows():
        i= [row['x'],row['y']]
        j=[centroids[row['closest']][0],centroids[row['closest']][1]]
        distance = calculate_distance(i,j)
        t=t+distance
    return t


    
""" running all 3 steps 
•	Initialisation - centroids initialization (from both strategy 1(get_centoroids) and 2(get_centoroids_2))
•	Assignment  -assignment of data points to near by centroids
•	Update -The centroid of the clusters becomes the new mean
Assignment and Update are repeated iteratively until convergence

The end result is that the sum of squared errors is minimised between points and their respective centroids.
"""
def run_startegy(df,clusters,strategy):
    wcs=[]
    for i in range(2,clusters+1):
        #choosing the centroid selection strategy
        if strategy=='S1':
            centroids=get_centoroids(df,i)
        else:
            centroids=get_centoroids_2(df,i)
        
        df = assignment(df, centroids)  
        #old_centroids = copy.deepcopy(centroids)
        
        #updating the centroids to near
        centroids = update(centroids)
        
        #running the logic till we get the final closest centroid, after which the centroid will not be changing
        while True:
            closest_centroids = df['closest'].copy(deep=True)
            centroids = update(centroids)
            df = assignment(df, centroids)
            if closest_centroids.equals(df['closest']):
                break
        #calculating optimaized value for the given centroids with the sample
        w=get_optimzed_values(df,centroids)
        wcs.append(w)
     
    return wcs

#plotting the figure with wcss and clusters value
def plot_graph(wcs,clusters,title):
    plt.xlabel("Number of Clusters")
    plt.ylabel("within-cluster sums of squares (WCSS)")
    plt.title(title)
    plt.plot(range(2,clusters+1),wcs)
    plt.figure()
    

    
#getting the data from the given dataset
df,colors=get_data()

#no of max clusters
clusters=10

wcs=run_startegy(df,clusters,'S1')
plot_graph(wcs,clusters,'with run_startegy_1')
print( 'With run_startegy_1 :',wcs)

wcs=run_startegy(df,clusters,'S1')
plot_graph(wcs,clusters,'with run_startegy_1_2')
print( 'With run_startegy_1_2 :',wcs)

wcs=run_startegy(df,clusters,'S2')
plot_graph(wcs,clusters,'with run_startegy_2')
print( 'With run_startegy_2 :',wcs)

wcs=run_startegy(df,clusters,'S2')
plot_graph(wcs,clusters,'with run_startegy_2_2')
print( 'With run_startegy_2_1 :',wcs)