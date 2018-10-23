#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 14:27:04 2018

@author: little_wings
"""
import fun_ware
import knn_ware
import minKL_ware
alpha=5
belta=1
C=40
K=15
#statsData,votesData,sideDic,statsWithVotesInfo = fun_ware.getDataWithVoteInfo()
#K=26
#trainDataLabel=[k for k in statsWithVotesInfo if k<=3000]
#testDataLabel=[k for k in statsWithVotesInfo if k>3000]
#
#diffKBindingMassDic = knn_ware.DSBestK(trainDataLabel,testDataLabel,statsData)
#KDic={k:diffKBindingMassDic[k][-1][0] if k in diffKBindingMassDic.keys() else 26 for k in testDataLabel }
#
#aimClassDic,neibourSamplesLabelDic,neibourBindingMassDic=knn_ware.knnClassWithDSRule(
#        trainDataLabel,testDataLabel,statsData,K,alpha=1,belta=1)

aimClassDicV,neibourSamplesLabelDicV,neibourBindingMassDicV=knn_ware.DSRuleWithVoteInfo(
        trainDataLabel,testDataLabel,statsData,votesData,K,alpha,belta,C)

#aimClassDicVDiffK,neibourSamplesLabelDicVDiffK,neibourBindingMassDicVDiffK=knn_ware.DSRuleWithVoteInfo(
#        trainDataLabel,testDataLabel,statsData,votesData,KDic,alpha=1,belta=1,C=1)
#
#neibourSamplesLabelDic,knnAimClassDic= knn_ware.knnClassMajorRule(
#    trainDataLabel,testDataLabel,statsData,K)
#
#
#neibourSamplesLabelDic,knnAimClassDicDiffK= knn_ware.knnClassMajorRule(
#    trainDataLabel,testDataLabel,statsData,KDic)
#
#aimClassDicDiffK,neibourSamplesLabelDicDiffK,neibourBindingMassDicDissK=knn_ware.knnClassWithDSRule(
#        trainDataLabel,testDataLabel,statsData,KDic,alpha=1,belta=1)
#
diffKBindingMassDicV = knn_ware.DSBestKWithVotes(trainDataLabel,testDataLabel,statsData,votesData,alpha,belta,C)
KDicV={k:diffKBindingMassDicV[k][-1][0] if k in diffKBindingMassDicV.keys() else 15 for k in testDataLabel }

aimClassDicDiffKV,neibourSamplesLabelDicDiffKV,neibourBindingMassDicDiffKV=knn_ware.DSRuleWithVoteInfo(
        trainDataLabel,testDataLabel,statsData,votesData,KDicV,alpha,belta,C)

error1=0
error2=0
error3=0
error4=0
error5=0
error6=0
error7=0

for label in testDataLabel:
    trueClass = statsData[label][12]
    
    if trueClass != aimClassDic[label].keys()[0]:
        error1+=1
    if trueClass != aimClassDicV[label].keys()[0]:
        error2+=1
    if trueClass != knnAimClassDic[label]:
        error3+=1
    if trueClass != knnAimClassDicDiffK[label]:
        error4+=1
    if trueClass !=aimClassDicVDiffK[label].keys()[0]:
        error5+=1
    if trueClass !=aimClassDicDiffK[label].keys()[0]:
        error6+=1
    if trueClass !=aimClassDicDiffKV[label].keys()[0]:
        error7+=1
    
