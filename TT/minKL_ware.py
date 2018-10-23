#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 19:18:14 2018

@author: little_wings
"""
import fun_ware
import knn_ware
import math
#statsData,votesData,sideDic,statsWithVotesInfo = fun_ware.getDataWithVoteInfo()
#
#K=26
#trainDataLabel=[k for k in statsWithVotesInfo if k<=3000]
#testDataLabel=[k for k in statsWithVotesInfo if k>3000]

def getSubPDicQDic(trainDataLabel,K,statsData):
    neibourSamplesLabelDic,aimClass = knn_ware.knnClassMajorRule(
                                        trainDataLabel,trainDataLabel,statsData,K)
    #PDic is the empirical class distribution
    PDic = {k: {label : len(labelC)/(K*1.0)  for label,labelC in v.items()} 
                for k,v in neibourSamplesLabelDic.items()}
     
    trueTrainLabelOfClass = {}
    for l in trainDataLabel:
        classL = statsData[l][12]
        if classL in trueTrainLabelOfClass:
            trueTrainLabelOfClass[classL].append(l)
        else:
            trueTrainLabelOfClass[classL]=[l]
    #QDic is the empirical centre class distribution
    QDic={}
    for j,v in trueTrainLabelOfClass.items():
        QDic[j]={}
        for label in v:
           for kk,prob in PDic[label].items():
               if kk in QDic[j]:
                   QDic[j][kk]+=prob
               else:
                   QDic[j][kk]=prob
        for k,p in QDic[j].items():
            QDic[j][k] = p / len(v)
    return PDic,QDic
                   
def getPDicQDic(trainDataLabel,K,statsData):
    neibourSamplesLabelDic,aimClass = knn_ware.knnClassMajorRule(
                                        trainDataLabel,trainDataLabel,statsData,K)
    PDic={}
    for k,neibourSet in neibourSamplesLabelDic.items():
        probDic={-1:0.0,1:0.0}
        for label,labelC in neibourSet.items():
            if label<0:
                probDic[-1] +=len(labelC)
            else:
                probDic[1] += len(labelC)
        probDic[-1]/=K
        probDic[1]/=K
        PDic[k] = probDic

    trueTrainLabelOfClass = {}
    for l in trainDataLabel:
        if statsData[l][12] < 0:
            classL = -1
        else:
            classL = 1
        if classL in trueTrainLabelOfClass:
            trueTrainLabelOfClass[classL].append(l)
        else:
            trueTrainLabelOfClass[classL]=[l]
    #QDic is the empirical centre class distribution
    QDic={}
    for j,v in trueTrainLabelOfClass.items():
        QDic[j]={}
        for label in v:
           for kk,prob in PDic[label].items():
               if kk in QDic[j]:
                   QDic[j][kk]+=prob
               else:
                   QDic[j][kk]=prob
        for k,p in QDic[j].items():
            QDic[j][k] = p / len(v)
    return PDic,QDic     
    
def minKLDivergence(trainDataLabel,testDataLabel,K,statsData):
    PPDic,QDic = getPDicQDic(trainDataLabel,K,statsData)
    PDic,QQDic=getPDicQDic(trainDataLabel+testDataLabel,K,statsData)
    KLDiDic={}
    aimClass={}
    for testLabel in testDataLabel:
        KLDiDic[testLabel]={}
        minKLD=999999
        aimC=0
        for j,vQ in QDic.items():
            KLDi = 0.0
            for i in PDic[testLabel].keys():
                pi=PDic[testLabel][i]
                qi=vQ[i]
                if pi < 0.0000001:
                    pi=0.0000001
                if qi <0.0000001:
                    qi=0.0000001
                KLDi += pi*math.log(pi/qi,2)
            KLDiDic[testLabel][j]=KLDi
            if minKLD > KLDi:
                minKLD = KLDi
                aimC=j
        aimClass[testLabel]=aimC
    return KLDiDic,aimClass
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    