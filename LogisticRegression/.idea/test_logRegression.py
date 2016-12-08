# -*- coding: utf-8 -*-
from numpy import *
from logRegression import *
import matplotlib.pyplot as plt
import time

def loadData():
    train_x=[]
    train_y=[]
    fileIn=open('/Users/chris/Documents/机器学习/LogisticRegression/testSet.txt')
    for line in fileIn.readlines():
        lineArr=line.strip().split()
        train_x.append([1.0,float(lineArr[0]),float(lineArr[1])])
        train_y.append(float(lineArr[2]))
    return mat(train_x),mat(train_y).transpose()

print 'step1:load data...'
train_x,train_y=loadData()
test_x=train_x
test_y=train_y

print 'step2:training...'
opts={'alpha':0.01,'maxIter':200,'optimizeType':'smoothStocGradDescent'}
optimalWeights=trainLogRegres(train_x,train_y,opts)

print 'step3:testing...'
accuracy=testLogRegres(optimalWeights,test_x,test_y)

print 'step4:show the result...'
print 'The classify accuracy is :%.3f%%'%(accuracy *100)
showLogRegres(optimalWeights,train_x,train_y)



