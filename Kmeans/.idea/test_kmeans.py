# -*-coding:utf-8 -*-
from numpy import *
import time
import matplotlib.pyplot as plt
import Kmeans

print 'step1:load data...'
dataSet=[]
fileIn=open('/Users/chris/Documents/机器学习/Kmeans/testSet.txt')
for line in fileIn.readlines():
    lineArr=line.strip().split('\t')
    dataSet.append([float(lineArr[0]),float(lineArr[1])])

print 'step2:cluster'
dataSet=mat(dataSet)
k=4
centroids,clusterAssment=Kmeans.biKmeans(dataSet,k)
print 'step3:show the result...'
Kmeans.showCluster(dataSet,k,centroids,clusterAssment)