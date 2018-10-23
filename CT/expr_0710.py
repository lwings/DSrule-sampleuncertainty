#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 20:00:36 2018

@author: little_wings
"""
import fun_ware
import knn_ware
import minKL_ware
statsData,votesData,sideDic,statsWithVotesInfo = fun_ware.getDataWithVoteInfo()

K=26
trainDataLabel=[k for k in statsWithVotesInfo if k<=3000]
testDataLabel=[k for k in statsWithVotesInfo if k>3000]

KLDiDic,aimClassDic=minKL_ware.minKLDivergence(trainDataLabel,
                                            testDataLabel,K,statsData)
neibourSamplesLabelDic,knnAimClassDic= knn_ware.knnClassMajorRule(
    trainDataLabel,testDataLabel,statsData,K)

error1=0
error2=0

for label in testDataLabel:
    trueClass = statsData[label][12]
    
    if trueClass * aimClassDic[label] < 0:
        error1+=1
    if trueClass * knnAimClassDic[label] < 0:
        error2+=1