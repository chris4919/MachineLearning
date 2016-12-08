# -*- coding: utf-8 -*-
from numpy import *
import SVM

print 'step1:loading data...'
dataSet=[]
labels=[]
fileIn=open("/Users/chris/Documents/机器学习/SVM/testSet.txt")
for line in fileIn.readlines():
    lineArr=line.strip().split('\t')
    dataSet.append([float(lineArr[0]),float(lineArr[1])])
    labels.append(float(lineArr[2]))

dataSet =mat(dataSet)
labels=mat(labels).T
train_x=dataSet[0:81,:]
train_y=labels[0:81,:]
test_x=dataSet[80:101,:]
test_y=labels[80:101,:]

print "step2:training...."
C=0.6
toler=0.001
maxIter=50
svmClassifier=SVM.trainSVM(train_x,train_y,C,toler,maxIter,kernelOption=('linear',0))

print "step3:testing..."
accuracy=SVM.testSVM(svmClassifier,test_x,test_y)

print "step4:show result"
print 'The classify accuracy is :%.3f%%' %(accuracy*100)
SVM.showSVM(svmClassifier)