#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 21:53:16 2018

@author: little_wings
"""

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
diffKBindingMassDicV = knn_ware.DSBestKWithVotes(trainDataLabel,testDataLabel,statsData,votesData,alpha,belta,C)
KDicV={k:diffKBindingMassDicV[k][-1][0] if k in diffKBindingMassDicV.keys() else 15 for k in testDataLabel }
#   
for K in range(5,36):
    statsData,votesData,sideDic,statsWithVotesInfo = fun_ware.getDataWithVoteInfo()
    
    trainDataLabel=[k for k in statsWithVotesInfo if k<=3000]
    testDataLabel=[k for k in statsWithVotesInfo if k>3000]
    
    
    aimClassDicV,neibourSamplesLabelDicV,neibourBindingMassDicV=knn_ware.DSRuleWithVoteInfo(
            trainDataLabel,testDataLabel,statsData,votesData,K,alpha,belta,C)
    
    
 
    neibourSamplesLabelDic,knnAimClassDic= knn_ware.knnClassMajorRule(
    trainDataLabel,testDataLabel,statsData,K)
    
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
        
        if trueClass != knnAimClassDic[label]:
            error3+=1    
        if trueClass != aimClassDicV[label].keys()[0]:
            error2+=1
        if trueClass !=aimClassDicDiffKV[label].keys()[0]:
            error7+=1
    print "error2"
    print error2-error7
    print "error3"
    print error3-error7
