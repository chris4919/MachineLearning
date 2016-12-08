# -*- coding: utf-8 -*-
from numpy import *
import operator
import os

# kNN算法
def kNNClassify(Input,dataSet,labels,k):
    numSamples=dataSet.shape[0]
#step1:计算距离
    diff=tile(Input,(numSamples,1))-dataSet
    squaredDiff=diff**2
    squaredDist=sum(squaredDiff,axis=1)
    distance=squaredDist**0.5

#step2:距离排序
    sortedDistIndices=argsort(distance)

    classCount={}
    for i in xrange(k):
        #step3:找K个最小距离
        votedLabel=labels[sortedDistIndices[i]]

        #step4:计算找出样本各自类别
        classCount[votedLabel]=classCount.get(voted Label,0)+1

    #step5:计算类别所占比例
    maxCount=0
    for key,value in classCount.items():
        if value > maxCount:
            maxCount = value
            maxIndex=key
    return maxIndex
#将图变成向量
def imgToVector(filename):
    rows=32
    cols=32
    imgVector=zeros((1,rows*cols))
    fileIn=open(filename)
    for row in xrange(rows):
        lineStr=fileIn.readline()
        for col in  xrange(cols):
            imgVector[0,row*32+col]=int(lineStr[col])
    return imgVector
#加载数据集
def loadDataSet():
    print "---Traning Set---"
    dataSetAddr='/Users/chris/Desktop/KNN/digits/'
    trainingFileList=os.listdir(dataSetAddr+'trainingDigits')
    numSamples=len(trainingFileList)

    train_x=zeros((numSamples,1024))
    train_y=[]
    for i in xrange(numSamples):
        filename=trainingFileList[i]

        train_x[i,:]=imgToVector(dataSetAddr +'trainingDigits/%s'% filename)
        label=int(filename.split('_')[0])
        train_y.append(label)
    print "---Testing Set---"
    testingFileList=os.listdir(dataSetAddr+'testDigits')
    numSamples =len(testingFileList)
    test_x=zeros((numSamples,1024))
    test_y=[]
    for i in xrange(numSamples):
        filename=testingFileList[i]
        test_x[i,:]=imgToVector(dataSetAddr+'testDigits/%s'%filename)
        label=int(filename.split('_')[0])
        test_y.append(label)
    return train_x,train_y,test_x,test_y

#测试
def testHandWritingClass():
    print "load data"
    train_x,train_y,test_x,test_y=loadDataSet()

    print "training"
    pass

    print "testing"
    numTestSamples=test_x.shape[0]
    matchCount=0
    for i in xrange(numTestSamples):
        predict=kNNClassify(test_x[i],train_x,train_y,3)
        if predict==test_y[i]:
            matchCount+=1
    accuracy=float(matchCount)/numTestSamples

    print "result: accuracy is :%.2f%%" %(accuracy*100)



