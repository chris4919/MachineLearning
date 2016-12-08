import KNN
from numpy import *
# -*- coding: utf-8 -*-

dataSet, labels=KNN.createDataSet()

test=array([1.2,1.0])
K=3
outputLabel = KNN.KNNClassify(test,dataSet,labels,K)
print "Input is :",test,"classified to class:",outputLabel

test=array([0.1,0.3])
outputLabel = KNN.KNNClassify(test,dataSet,labels,K)
print "Input is :",test,"classified to class:",outputLabel
