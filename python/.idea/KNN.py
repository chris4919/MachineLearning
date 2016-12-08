# -*- coding: utf-8 -*-
from numpy import *
import operator


def createDataSet():
    # 创建一个样例矩阵
    group = array([[1.0,0.9],[1.0,1.0],[0.1,0.2],[0.0,0.1]])
    labels=['A','A','B','B']
    return group,labels

#KNN算法
def KNNClassify(Input,dataSet,labels,k):
    numSamples=dataSet.shape[0]

   #step1 算距离
    diff=tile(Input,(numSamples,1))-dataSet
    squaredDiff=diff**2
    squaredDist=sum(squaredDiff,axis=1)
    distance = squaredDist**0.5

    #step2 距离排序
    sortedDistIndices = argsort(distance)

    classCount={}
    for i in xrange(k):
        #step3 选择前K个最小距离
        voteLabel=labels[sortedDistIndices[i]]

        #step4 计算K个临近样本所属类别
        classCount[voteLabel]=classCount.get(voteLabel,0)+1

    #step5 返回占比例最高的类别
    maxCount=0
    for key,value in classCount.items():
        if value>maxCount:
            maxCount=value
            maxIndex=key
    return maxIndex








