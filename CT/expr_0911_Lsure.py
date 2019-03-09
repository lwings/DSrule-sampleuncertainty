#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 16:08:35 2018

@author: little_wings
"""

import fun_ware
import knn_ware
import matplotlib.pyplot as plt
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
testDataLabel2=[k for k in statsWithVotesInfo if k>3500 and k<34000]
alpha=1
belta=1
C=1
Kmin=10
Kmax=20
errorList1=[]
errorList2=[]
errorList3=[]
errorList4=[]
errorList5=[]
for L in range(1,18):
#    BestK,minErrors = knn_ware.globalDSBestK(Kmin,Kmax,trainDataLabel,testDataLabel,statsData,votesData,alpha,belta,C) 
#    BestK2,minErrors = knn_ware.globalLsureDSBestK(Kmin,Kmax,trainDataLabel,testDataLabel,statsData,votesData,L,alpha,belta,C) 

    
    
    aimClassDicV,neibourSamplesLabelDicV,neibourBindingMassDicV=knn_ware.DSRuleWithVoteInfo(
            trainDataLabel,testDataLabel2,statsData,votesData,40,alpha,belta,C)
    aimClassDic,neibourSamplesLabelDic,neibourBindingMassDic=knn_ware. DSRuleWithLSureVoteInfo(
            trainDataLabel,testDataLabel2,statsData,votesData,40,L,alpha,belta,C)
#    aimClassDic2,neibourSamplesLabelDic2,neibourBindingMassDic2=knn_ware. DSRuleLocalWeightLSureVoteInfo(
#            trainDataLabel,testDataLabel2,statsData,votesData,40,L,alpha,belta,C)
#    aimClassDic3,neibourSamplesLabelDic3,neibourBindingMassDic3=knn_ware. DSRuleLocalWeightLSureVoteInfoWithGlobalWeight(
#            trainDataLabel,testDataLabel2,statsData,votesData,40,L,alpha,belta,C)
#    aimClassDic4,neibourSamplesLabelDic4,neibourBindingMassDic4=knn_ware. DSRuleLocalBayesianWeightLSureVoteInfo(
#            trainDataLabel,testDataLabel2,statsData,votesData,40,L,alpha,belta,C)
    
    error1=0
    error2=0
    error3=0
    error4=0
    error5=0
    for label in testDataLabel2:
        if aimClassDicV[label].keys()[0] != statsData[label][12]:
            error1+=1
        if aimClassDic[label].keys()[0] != statsData[label][12]:
            error2+=1
#        if aimClassDic2[label].keys()[0] != statsData[label][12]:
#            error3+=1
#        if aimClassDic3[label].keys()[0] != statsData[label][12]:
#            error4+=1
#        if aimClassDic4[label].keys()[0] != statsData[label][12]:
#            error5+=1

    print(L)   
    print(error1)
    print(error2)
#    print(error3)
#    print(error4)
#    print(error5)
    errorList1.append(error1)
    errorList2.append(error2)
#    errorList3.append(error3)
#    errorList4.append(error4)
#    errorList5.append(error5)
          

x=range(1,18)
plt.plot(x,errorList1,color="red")
plt.plot(x,errorList2,color="blue")
#plt.plot(x,errorList3,color="green")
#plt.plot(x,errorList4,color="yellow")
#plt.plot(x,errorList5,color="black")
