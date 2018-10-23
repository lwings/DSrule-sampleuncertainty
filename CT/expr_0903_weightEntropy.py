#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 16:57:20 2018

@author: little_wings
"""

import fun_ware
import knn_ware

statsData,votesData,sideDic,statsWithVotesInfo = fun_ware.getDataWithVoteInfo()
#one side case
#statsData: 0-11 digits_____label 
#           12 digit________class
#           13 digit________side
#votesData: 0____side
#            1____user_id
#            2____firstvote
#            3____ultimavote
votesDataB={k:v for k,v in votesData.items() if k in statsWithVotesInfo}
trainDataLabel=[k for k in statsWithVotesInfo if k<=3000]
testDataLabel=[k for k in statsWithVotesInfo if k>3000 and k<3500] 
alpha=1
belta=1
C=1
Kmin=6
Kmax=36
BestK,minErrors = knn_ware.globalDSBestK(Kmin,Kmax,trainDataLabel,testDataLabel,statsData,votesData,alpha,belta,C) 
BestK2,minErrors = knn_ware.globalWeiDSBestK(Kmin,Kmax,trainDataLabel,testDataLabel,statsData,votesDataB,alpha,belta,C) 
testDataLabel2=[k for k in statsWithVotesInfo if k>3500 and k<9999]

#diffKBindingMassDic = knn_ware.dynamicDSBestKWithVotes(statsWithVotesInfo,testDataLabel2,statsData,votesData,alpha,belta,C)
#KDicV={k:diffKBindingMassDic[k][-1][0] if k in diffKBindingMassDic.keys() else 15 for k in testDataLabel2 }
aimClassDicV,neibourSamplesLabelDicV,neibourBindingMassDicV=knn_ware.DSRuleWithVoteInfo(
        trainDataLabel,testDataLabel2,statsData,votesData,BestK,alpha,belta,C)
aimClassDic,neibourSamplesLabelDic,neibourBindingMassDic=knn_ware.DSRuleWithWeightedVoteInfo(
        trainDataLabel,testDataLabel2,statsData,votesDataB,BestK2,alpha,belta,C)

error1=0
error2=0

for label in testDataLabel2:
    if aimClassDicV[label].keys()[0] != statsData[label][12]:
        error1+=1
    if aimClassDic[label].keys()[0] != statsData[label][12]:
        error2+=1
        
        
        
        
