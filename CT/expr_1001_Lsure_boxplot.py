#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 20:33:43 2018

@author: little_wings
"""

import fun_ware
import knn_ware
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
totalDoctor=14
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
#testDataLabel=[k for k in statsWithVotesInfo if k>3000 and k<3500] 
#testDataLabel2=[k for k in statsWithVotesInfo if k>3500 and k<3000]
kerror1=[]
kerror2=[]
kerror3=[]
for k in range(5):
    testDataLabel2=statsWithVotesInfo[1028+k*30:1028+200+k*30]
    kerror2.append([]);
    kerror1.append([]);
    kerror3.append([]);
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
    for L in range(1,totalDoctor):
        #    BestK,minErrors = knn_ware.globalDSBestK(Kmin,Kmax,trainDataLabel,testDataLabel,statsData,votesData,alpha,belta,C) 
        #    BestK2,minErrors = knn_ware.globalLsureDSBestK(Kmin,Kmax,trainDataLabel,testDataLabel,statsData,votesData,L,alpha,belta,C) 
        
            
            
            aimClassDicV,neibourSamplesLabelDicV,neibourBindingMassDicV=knn_ware.DSRuleWithVoteInfo(
                    trainDataLabel,testDataLabel2,statsData,votesData,40,alpha,belta,C)
#            aimClassDic,neibourSamplesLabelDic,neibourBindingMassDic=knn_ware. DSRuleWithLSureVoteInfo(
#                    trainDataLabel,testDataLabel2,statsData,votesData,40,L,alpha,belta,C)
            aimClassDic2,neibourSamplesLabelDic2,neibourBindingMassDic2=knn_ware. DSRuleLocalWeightLSureVoteInfo(
                    trainDataLabel,testDataLabel2,statsData,votesData,40,L,alpha,belta,C)
#            aimClassDic3,neibourSamplesLabelDic3,neibourBindingMassDic3=knn_ware. DSRuleLocalWeightLSureVoteInfoWithGlobalWeight(
#                    trainDataLabel,testDataLabel2,statsData,votesData,40,L,alpha,belta,C)
#            aimClassDic4,neibourSamplesLabelDic4,neibourBindingMassDic4=knn_ware. DSRuleLocalBayesianWeightLSureVoteInfo(
#                    trainDataLabel,testDataLabel2,statsData,votesData,40,L,alpha,belta,C)
            error1=0
            error2=0
            error3=0
            error4=0
            error5=0
            for label in testDataLabel2:
                if aimClassDicV[label].keys()[0] != statsData[label][12]:
                    error1+=1
#                if aimClassDic[label].keys()[0] != statsData[label][12]:
#                    error2+=1
                if aimClassDic2[label].keys()[0] != statsData[label][12]:
                    error3+=1
        #        if aimClassDic3[label].keys()[0] != statsData[label][12]:
        #            error4+=1
        #        if aimClassDic4[label].keys()[0] != statsData[label][12]:
        #            error5+=1
        
#            print(L)   
#            print(error1)
#            print(error2)
        #    print(error3)
        #    print(error4)
        #    print(error5)
#            errorList1.append(error1)
#            errorList2.append(error2)
            kerror1[k].append(error1)
#            kerror2[k].append(error2)
            kerror3[k].append(error3)
        #    errorList3.append(error3)
        #    errorList4.append(error4)
        #    errorList5.append(error5)
                  
        
#    x=range(1,totalDoctor)
#    plt.plot(x,errorList1,color="red")
#    plt.plot(x,errorList2,color="blue")
    #plt.plot(x,errorList3,color="green")
    #plt.plot(x,errorList4,color="yellow")
    #plt.plot(x,errorList5,color="black")
    
    
    
    
    
    
    
    
#plot
dferror2=pd.DataFrame()
dferror1=pd.DataFrame()
dferror3=pd.DataFrame()
for i in range(totalDoctor-1):
#    dferror2[i]=[kerror2[k][i] for k in range(len(kerror1))]
    dferror1[i]=[kerror1[k][i] for k in range(len(kerror1))]
    dferror3[i]=[kerror3[k][i] for k in range(len(kerror1))]
#plt.boxplot(x=dferror2.values,labels=dferror2.columns,whis=1.5)
plt.boxplot(x=dferror1.values,labels=dferror1.columns,whis=4)
plt.boxplot(x=dferror3.values,labels=dferror1.columns,whis=4)
plt.show()
