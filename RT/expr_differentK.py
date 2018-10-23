#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 21:29:37 2018

@author: little_wings
"""
import fun_ware
import knn_ware
import operator

statsData,votesData,sideDic,statsWithVotesInfo = fun_ware.getDataWithVoteInfo()
#one side case
#statsData: 0-11 digits_____label 
#           12 digit________class
#           13 digit________side
#votesData: 0____side
#            1____user_id
#            2____firstvote
#            3____ultimavote

trainDataLabel=[k for k in statsWithVotesInfo if k<=3000]
testDataLabel=[k for k in statsWithVotesInfo if k>3000] 


diffKBindingMassDic={}
for K in range(5,36):
    print K
    aimClassDic,neibourSamplesLabelDic,neibourBindingMassDic = knn_ware.knnClassWithDSRule(
        trainDataLabel,testDataLabel,statsData,K)
    massDecisionDic={k:v[-1][1]-v[-2][1] for k,v in neibourBindingMassDic.items() if len(v) >=2}
    
    for k,v in massDecisionDic.items():
        if k in diffKBindingMassDic:
            diffKBindingMassDic[k][K]=massDecisionDic[k]
        else:
            diffKBindingMassDic[k]={K:massDecisionDic[k]}

for k,v in diffKBindingMassDic.items():
    if len(diffKBindingMassDic[k])>1:
        diffKBindingMassDic[k] = sorted(diffKBindingMassDic[k].items(),key=operator.itemgetter(1))

for k,v in diffKBindingMassDic.items():
    
    print v[-1][0]