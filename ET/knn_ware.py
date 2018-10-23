#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 15:08:08 2018

@author: little_wings
"""
import operator
import math
import fun_ware
def massCompute(a,b,alpha=1,belta=1,entropy=0,C=1):
#statsData: 0-11 digits_____label 
#           12 digit________class
#           13 digit________side
    
# a and b are both lists
    d=0.0
    for i in range(0,12):
        if a[i]==0 or b[i]==0:
            d+=0.5
        elif a[i] != b[i]:
            d+=1.0
#            d+=(a[i]-b[i])*0.1-0.1
    return alpha*math.exp(-1*belta*d)*math.exp(-1*entropy*C)


def distanceCompute(a,b):
#statsData: 0-11 digits_____label 
#           12 digit________class
#           13 digit________side
    
# a and b are both lists
    d=0.0
    for i in range(0,12):
        if a[i]==0 or b[i]==0:
            d+=0.5
        elif a[i] != b[i]:
            d+=1.0
#            d+=(a[i]-b[i])*0.1-0.1
        
    return d


def DSEvidenceCombine(sampleNeibourMass):
#primitive method(simple frame of discernment)
#sampleNeibourMass is a dictionary, it's for one point
    sampleBindingMass={}
#    compute the sampleBindingMass:
    for k,v in sampleNeibourMass.items():
        sumF = 1.0
        for listItr in v:
            sumF*= 1-listItr[1]
        sampleBindingMass[k] = 1 - sumF
#   compute the normalizing factor K:
    K1=0.0
    K2=1.0
    for k,v in sampleBindingMass.items():
        K2 *= 1.0-v 
        k1t=1.0
        for kt,vt in sampleBindingMass.items():
            if(kt==k):
                k1t*=vt
            else:
                k1t*=1.0-vt
        K1+=k1t
    K=K1+K2
#    compute the binding mass function:
    bindingMassDic={}
    for k,v in sampleBindingMass.items():
        res=1.0
        for kt,vt in sampleBindingMass.items():
            if(kt==k):
                res*=vt
            else:
                res*=1.0-vt
        bindingMassDic[k]=res/K
    bindingMassDic=sorted(bindingMassDic.items(),key=operator.itemgetter(1))
    aimClass={bindingMassDic[-1][0]:bindingMassDic[-1][1]}
    return aimClass,bindingMassDic


def kNeibours(trainDataLabel,sampleLabel,statsData,K):
    if type(K) is dict:
        K=K[sampleLabel]
    distanceDic={}
    for seq in trainDataLabel:
        if seq!= sampleLabel:
            distance = distanceCompute(statsData[sampleLabel],statsData[seq])
            distanceDic[seq]=distance
    sorted_distanceDic=sorted(distanceDic.items(),key=operator.itemgetter(1))
    sampleNeibourLabels=[sorted_distanceDic[k][0] for k in range(K)]
    return sampleNeibourLabels


def knnClassMajorRule(trainDataLabel,testDataLabel,statsData,K): 
    neibourSamplesLabelDic={}
    aimClassDic={}
    for seq in testDataLabel:
        sampleNeibourLabels=kNeibours(trainDataLabel,seq,statsData,K)
        neibourDic={}
        for s in sampleNeibourLabels:
            CTScheme=statsData[s][12]
            if CTScheme in neibourDic:
                neibourDic[CTScheme].append(statsData[s])
            else:
                neibourDic[CTScheme]=[statsData[s]]   
        maxvalue=0
        for k,v in neibourDic.items():
            if len(v) > maxvalue:
                maxvalue=len(v)
                aimClass=k
        aimClassDic[seq]=aimClass
        neibourSamplesLabelDic[seq]=neibourDic    
        
    return neibourSamplesLabelDic,aimClassDic
    

def knnClassWithDSRule(trainDataLabel,testDataLabel,
                       statsData,K,alpha=1,belta=1):
    neibourSamplesLabelDic={}
    aimClassDic={}
    neibourSamplesMassDic={}
    neibourBindingMassDic={}
    for seq in testDataLabel:
        sampleNeibourLabels=kNeibours(trainDataLabel,seq,statsData,K)
#        sampleNeibourMass=[massCompute(a=statsData[seq],b=statsData[seq])
#        for sa in sampleNeibourLabels]
        sampleNeibourMass={}
        for sa in sampleNeibourLabels:
#            sampleNeibourMass[statsData[sa][12]]=[sa,
#                              massCompute(a=statsData[seq],b=statsData[sa])]
            CTScheme=statsData[sa][12]
            if CTScheme in sampleNeibourMass:
                sampleNeibourMass[CTScheme].append([sa,
                                 massCompute(statsData[seq],statsData[sa],
                                             alpha,belta,C=1)])
            else:
                sampleNeibourMass[CTScheme]=[[sa,
                                 massCompute(statsData[seq],statsData[sa],
                                             alpha,belta,C=1)]]
        neibourSamplesMassDic[seq]=sampleNeibourMass
        aimClass,neibourBindingMassDic[seq]=DSEvidenceCombine(sampleNeibourMass)
        neibourDic={}
        for s in sampleNeibourLabels:
            CTScheme=statsData[s][12]
            if CTScheme in neibourDic:
                neibourDic[CTScheme].append(statsData[s])
            else:
                neibourDic[CTScheme]=[statsData[s]]  
        neibourSamplesLabelDic[seq]=neibourDic
        aimClassDic[seq]=aimClass
    return aimClassDic,neibourSamplesLabelDic,neibourBindingMassDic
        
        
def DSRuleWithVoteInfo(trainDataLabel,testDataLabel,statsData,
                       votesData,K,alpha=1,belta=1,C=1):
    
    firstEntropyDic,UltimaEntropyDic = fun_ware.voteEntropyComp(votesData)
    
    neibourSamplesLabelDic={}
    aimClassDic={}
    neibourSamplesMassDic={}
    neibourBindingMassDic={}
    for seq in testDataLabel:
        sampleNeibourLabels=kNeibours(trainDataLabel,seq,statsData,K)
        sampleNeibourMass={}
        for sa in sampleNeibourLabels:
            CTScheme=statsData[sa][12]
            if CTScheme in sampleNeibourMass:
                sampleNeibourMass[CTScheme].append([sa,
                                 massCompute(statsData[seq],statsData[sa],
                                             alpha,belta,firstEntropyDic[sa],C)])
            else:
                sampleNeibourMass[CTScheme]=[[sa,
                                 massCompute(statsData[seq],statsData[sa],
                                             alpha,belta,firstEntropyDic[sa],C)]]
        neibourSamplesMassDic[seq]=sampleNeibourMass
        aimClass,neibourBindingMassDic[seq]=DSEvidenceCombine(sampleNeibourMass)
        neibourDic={}
        for s in sampleNeibourLabels:
            CTScheme=statsData[s][12]
            if CTScheme in neibourDic:
                neibourDic[CTScheme].append(statsData[s])
            else:
                neibourDic[CTScheme]=[statsData[s]]  
        neibourSamplesLabelDic[seq]=neibourDic
        aimClassDic[seq]=aimClass
    return aimClassDic,neibourSamplesLabelDic,neibourBindingMassDic

def DSRuleWithWeightedVoteInfo(trainDataLabel,testDataLabel,statsData,
                       votesData,K,alpha=1,belta=1,C=1):
    
    firstEntropyDic,UltimaEntropyDic = fun_ware.voteWeightedEntropyComp(votesData)
    
    neibourSamplesLabelDic={}
    aimClassDic={}
    neibourSamplesMassDic={}
    neibourBindingMassDic={}
    for seq in testDataLabel:
        sampleNeibourLabels=kNeibours(trainDataLabel,seq,statsData,K)
        sampleNeibourMass={}
        for sa in sampleNeibourLabels:
            CTScheme=statsData[sa][12]
            if CTScheme in sampleNeibourMass:
                sampleNeibourMass[CTScheme].append([sa,
                                 massCompute(statsData[seq],statsData[sa],
                                             alpha,belta,firstEntropyDic[sa],C)])
            else:
                sampleNeibourMass[CTScheme]=[[sa,
                                 massCompute(statsData[seq],statsData[sa],
                                             alpha,belta,firstEntropyDic[sa],C)]]
        neibourSamplesMassDic[seq]=sampleNeibourMass
        aimClass,neibourBindingMassDic[seq]=DSEvidenceCombine(sampleNeibourMass)
        neibourDic={}
        for s in sampleNeibourLabels:
            CTScheme=statsData[s][12]
            if CTScheme in neibourDic:
                neibourDic[CTScheme].append(statsData[s])
            else:
                neibourDic[CTScheme]=[statsData[s]]  
        neibourSamplesLabelDic[seq]=neibourDic
        aimClassDic[seq]=aimClass
    return aimClassDic,neibourSamplesLabelDic,neibourBindingMassDic
def DSBestK(trainDataLabel,testDataLabel,statsData,alpha=1,belta=1):
    diffKBindingMassDic={}
    for K in range(5,36):
        aimClassDic,neibourSamplesLabelDic,neibourBindingMassDic = knnClassWithDSRule(
            trainDataLabel,testDataLabel,statsData,K,alpha,belta)
        massDecisionDic={k:v[-1][1]-v[-2][1] for k,v in neibourBindingMassDic.items() if len(v) >=2}
        
        for k,v in massDecisionDic.items():
            if k in diffKBindingMassDic:
                diffKBindingMassDic[k][K]=massDecisionDic[k]
            else:
                diffKBindingMassDic[k]={K:massDecisionDic[k]}
    
    for k,v in diffKBindingMassDic.items():
        if len(diffKBindingMassDic[k])>1:
            diffKBindingMassDic[k] = sorted(diffKBindingMassDic[k].items(),key=operator.itemgetter(1))
    return diffKBindingMassDic
        
def DSBestKWithVotes(trainDataLabel,testDataLabel,statsData,votesData,alpha=1,belta=1,C=1):
    diffKBindingMassDic={}
    for K in range(5,36):
        aimClassDic,neibourSamplesLabelDic,neibourBindingMassDic = DSRuleWithVoteInfo(
        trainDataLabel,testDataLabel,statsData,votesData,K,alpha,belta,C)
        massDecisionDic={k:v[-1][1] for k,v in neibourBindingMassDic.items() if len(v) >=2}
        
        for k,v in massDecisionDic.items():
            if k in diffKBindingMassDic: 
                diffKBindingMassDic[k][K]=massDecisionDic[k]
            else:
                diffKBindingMassDic[k]={K:massDecisionDic[k]}
    
    for k,v in diffKBindingMassDic.items():
        if len(diffKBindingMassDic[k])>1:
            diffKBindingMassDic[k] = sorted(diffKBindingMassDic[k].items(),key=operator.itemgetter(1))
    return diffKBindingMassDic
        
def BestKForNewSample(statsWithVotesInfo,sampleLabel,statsData,votesData,alpha=1,belta=1,C=1):
    if type(sampleLabel) is int:
        sampleLabel = [sampleLabel]
    trainDataLabel=[label for label in statsWithVotesInfo if label < sampleLabel]
    diffKBindingMassDic={}
    for K in range(5,36):
        aimClassDic,neibourSamplesLabelDic,neibourBindingMassDic = DSRuleWithVoteInfo(
        trainDataLabel,sampleLabel,statsData,votesData,K,alpha,belta,C)
        massDecisionDic={k:v[-1][1] for k,v in neibourBindingMassDic.items() if len(v) >=2}
        
        for k,v in massDecisionDic.items():
            if k in diffKBindingMassDic:
                diffKBindingMassDic[k][K]=massDecisionDic[k]
            else:
                diffKBindingMassDic[k]={K:massDecisionDic[k]}
    
    for k,v in diffKBindingMassDic.items():
        if len(diffKBindingMassDic[k])>1:
            diffKBindingMassDic[k] = sorted(diffKBindingMassDic[k].items(),key=operator.itemgetter(1))
    return diffKBindingMassDic
            
def dynamicDSBestKWithVotes(statsWithVotesInfo,testDataLabel,statsData,votesData,alpha=1,belta=1,C=1):
    diffKBindingMassDic={}
    for sampleLabel in  testDataLabel:
        if len(BestKForNewSample(statsWithVotesInfo,sampleLabel,statsData,votesData,alpha,belta,C).values())>0 :
            diffKBindingMassDic[sampleLabel] = BestKForNewSample(
                    statsWithVotesInfo,sampleLabel,statsData,votesData,alpha,belta,C).values()[0]
    return diffKBindingMassDic
def weightedBestKForNewSample(statsWithVotesInfo,sampleLabel,statsData,votesData,alpha=1,belta=1,C=1):
    if type(sampleLabel) is int:
        sampleLabel = [sampleLabel]
    trainDataLabel=[label for label in statsWithVotesInfo if label < sampleLabel]
    diffKBindingMassDic={}
    for K in range(5,36):
        aimClassDic,neibourSamplesLabelDic,neibourBindingMassDic = DSRuleWithWeightedVoteInfo(
        trainDataLabel,sampleLabel,statsData,votesData,K,alpha,belta,C)
        massDecisionDic={k:v[-1][1] for k,v in neibourBindingMassDic.items() if len(v) >=2}
        
        for k,v in massDecisionDic.items():
            if k in diffKBindingMassDic:
                diffKBindingMassDic[k][K]=massDecisionDic[k]
            else:
                diffKBindingMassDic[k]={K:massDecisionDic[k]}
    
    for k,v in diffKBindingMassDic.items():
        if len(diffKBindingMassDic[k])>1:
            diffKBindingMassDic[k] = sorted(diffKBindingMassDic[k].items(),key=operator.itemgetter(1))
    return diffKBindingMassDic
            
def dynamicDSBestKWithWeightedVotes(statsWithVotesInfo,testDataLabel,statsData,votesData,alpha=1,belta=1,C=1):
    diffKBindingMassDic={}
    for sampleLabel in  testDataLabel:
        if len(BestKForNewSample(statsWithVotesInfo,sampleLabel,statsData,votesData,alpha,belta,C).values())>0 :
            diffKBindingMassDic[sampleLabel] = weightedBestKForNewSample(
                    statsWithVotesInfo,sampleLabel,statsData,votesData,alpha,belta,C).values()[0]
    return diffKBindingMassDic
            
def globalDSBestK(Kmin,Kmax,trainDataLabel,testDataLabel,statsData,votesData,alpha=1,belta=1,C=1):
    BestK=-1
    minErrors=9999
    for K in range(Kmin,Kmax):
        aimClassDic,neibourSamplesLabelDic,neibourBindingMassDic = DSRuleWithVoteInfo(trainDataLabel,testDataLabel,statsData,
                       votesData,K,alpha,belta,C)
        error=0
        for label,aimclass in aimClassDic.items():
            trueClass = statsData[label][12]
            if aimclass.keys()[0] != trueClass:
                error+=1
        if error<minErrors:
            minErrors = error
            BestK = K
    
    return BestK,minErrors           
            
def globalWeiDSBestK(Kmin,Kmax,trainDataLabel,testDataLabel,statsData,votesData,alpha=1,belta=1,C=1):
    BestK=-1
    minErrors=9999
    for K in range(Kmin,Kmax):
        aimClassDic,neibourSamplesLabelDic,neibourBindingMassDic = DSRuleWithWeightedVoteInfo(trainDataLabel,testDataLabel,statsData,
                       votesData,K,alpha,belta,C)
        error=0
        for label,aimclass in aimClassDic.items():
            trueClass = statsData[label][12]
            if aimclass.keys()[0] != trueClass:
                error+=1
        if error<minErrors:
            minErrors = error
            BestK = K
    
    return BestK,minErrors      
     
def globalLsureDSBestK(Kmin,Kmax,trainDataLabel,testDataLabel,statsData,votesData,L,alpha=1,belta=1,C=1):
    BestK=-1
    minErrors=9999
    for K in range(Kmin,Kmax):
        aimClassDic,neibourSamplesLabelDic,neibourBindingMassDic = DSRuleWithLSureVoteInfo(trainDataLabel,testDataLabel,statsData,
                       votesData,K,L,alpha,belta,C)
        error=0
        for label,aimclass in aimClassDic.items():
            trueClass = statsData[label][12]
            if aimclass.keys()[0] != trueClass:
                error+=1
        if error<minErrors:
            minErrors = error
            BestK = K
    
    return BestK,minErrors 

def DSRuleWithLSureVoteInfo(trainDataLabel,testDataLabel,statsData,
                       votesData,K,L,alpha=1,belta=1,C=1):
    
    firstEntropyDic,UltimaEntropyDic = fun_ware.voteLsureEntropyComp(votesData,L)
    
    neibourSamplesLabelDic={}
    aimClassDic={}
    neibourSamplesMassDic={}
    neibourBindingMassDic={}
    for seq in testDataLabel:
        sampleNeibourLabels=kNeibours(trainDataLabel,seq,statsData,K)
        sampleNeibourMass={}
        for sa in sampleNeibourLabels:
            CTScheme=statsData[sa][12]
            if CTScheme in sampleNeibourMass:
                sampleNeibourMass[CTScheme].append([sa,
                                 massCompute(statsData[seq],statsData[sa],
                                             alpha,belta,firstEntropyDic[sa],C)])
            else:
                sampleNeibourMass[CTScheme]=[[sa,
                                 massCompute(statsData[seq],statsData[sa],
                                             alpha,belta,firstEntropyDic[sa],C)]]
        neibourSamplesMassDic[seq]=sampleNeibourMass
        aimClass,neibourBindingMassDic[seq]=DSEvidenceCombine(sampleNeibourMass)
        neibourDic={}
        for s in sampleNeibourLabels:
            CTScheme=statsData[s][12]
            if CTScheme in neibourDic:
                neibourDic[CTScheme].append(statsData[s])
            else:
                neibourDic[CTScheme]=[statsData[s]]  
        neibourSamplesLabelDic[seq]=neibourDic
        aimClassDic[seq]=aimClass
    return aimClassDic,neibourSamplesLabelDic,neibourBindingMassDic
            
def DSRuleLocalWeightLSureVoteInfo(trainDataLabel,testDataLabel,statsData,
                       votesData,K,L,alpha=1,belta=1,C=1):
    
#    firstEntropyDic,UltimaEntropyDic = fun_ware.voteLsureEntropyComp(votesData,L)
    
    neibourSamplesLabelDic={}
    aimClassDic={}
    neibourSamplesMassDic={}
    neibourBindingMassDic={}
    for seq in testDataLabel:
        sampleNeibourLabels=kNeibours(trainDataLabel,seq,statsData,K)
        localVotesData={k:votesData[k] for k in sampleNeibourLabels}
        firstEntropyDic = fun_ware.localWeightLsureEntropyComp(localVotesData,L)
        sampleNeibourMass={}
        for sa in sampleNeibourLabels:
            CTScheme=statsData[sa][12]
            if CTScheme in sampleNeibourMass:
                sampleNeibourMass[CTScheme].append([sa,
                                 massCompute(statsData[seq],statsData[sa],
                                             alpha,belta,firstEntropyDic[sa],C)])
            else:
                sampleNeibourMass[CTScheme]=[[sa,
                                 massCompute(statsData[seq],statsData[sa],
                                             alpha,belta,firstEntropyDic[sa],C)]]
        neibourSamplesMassDic[seq]=sampleNeibourMass
        aimClass,neibourBindingMassDic[seq]=DSEvidenceCombine(sampleNeibourMass)
        neibourDic={}
        for s in sampleNeibourLabels:
            CTScheme=statsData[s][12]
            if CTScheme in neibourDic:
                neibourDic[CTScheme].append(statsData[s])
            else:
                neibourDic[CTScheme]=[statsData[s]]  
        neibourSamplesLabelDic[seq]=neibourDic
        aimClassDic[seq]=aimClass
    return aimClassDic,neibourSamplesLabelDic,neibourBindingMassDic            
            
            
def DSRuleLocalWeightLSureVoteInfoWithGlobalWeight(trainDataLabel,testDataLabel,statsData,
                       votesData,K,L,alpha=1,belta=1,C=1):
    
#    firstEntropyDic,UltimaEntropyDic = fun_ware.voteLsureEntropyComp(votesData,L)
    
    neibourSamplesLabelDic={}
    aimClassDic={}
    neibourSamplesMassDic={}
    neibourBindingMassDic={}
    for seq in testDataLabel:
        sampleNeibourLabels=kNeibours(trainDataLabel,seq,statsData,K)
        localVotesData={k:votesData[k] for k in sampleNeibourLabels}
        firstEntropyDic = fun_ware. localWeightLsureEntropyCompWithGlobalWeight(localVotesData,L,votesData)
        sampleNeibourMass={}
        for sa in sampleNeibourLabels:
            CTScheme=statsData[sa][12]
            if CTScheme in sampleNeibourMass:
                sampleNeibourMass[CTScheme].append([sa,
                                 massCompute(statsData[seq],statsData[sa],
                                             alpha,belta,firstEntropyDic[sa],C)])
            else:
                sampleNeibourMass[CTScheme]=[[sa,
                                 massCompute(statsData[seq],statsData[sa],
                                             alpha,belta,firstEntropyDic[sa],C)]]
        neibourSamplesMassDic[seq]=sampleNeibourMass
        aimClass,neibourBindingMassDic[seq]=DSEvidenceCombine(sampleNeibourMass)
        neibourDic={}
        for s in sampleNeibourLabels:
            CTScheme=statsData[s][12]
            if CTScheme in neibourDic:
                neibourDic[CTScheme].append(statsData[s])
            else:
                neibourDic[CTScheme]=[statsData[s]]  
        neibourSamplesLabelDic[seq]=neibourDic
        aimClassDic[seq]=aimClass
    return aimClassDic,neibourSamplesLabelDic,neibourBindingMassDic                  
            
def DSRuleLocalBayesianWeightLSureVoteInfo(trainDataLabel,testDataLabel,statsData,
                       votesData,K,L,alpha=1,belta=1,C=1):
    
#    firstEntropyDic,UltimaEntropyDic = fun_ware.voteLsureEntropyComp(votesData,L)
    
    neibourSamplesLabelDic={}
    aimClassDic={}
    neibourSamplesMassDic={}
    neibourBindingMassDic={}
    for seq in testDataLabel:
        sampleNeibourLabels=kNeibours(trainDataLabel,seq,statsData,K)
        localVotesData={k:votesData[k] for k in sampleNeibourLabels}
        firstEntropyDic = fun_ware.localBayesianWeightLsureEntropyComp(localVotesData,L)
        sampleNeibourMass={}
        for sa in sampleNeibourLabels:
            CTScheme=statsData[sa][12]
            if CTScheme in sampleNeibourMass:
                sampleNeibourMass[CTScheme].append([sa,
                                 massCompute(statsData[seq],statsData[sa],
                                             alpha,belta,firstEntropyDic[sa],C)])
            else:
                sampleNeibourMass[CTScheme]=[[sa,
                                 massCompute(statsData[seq],statsData[sa],
                                             alpha,belta,firstEntropyDic[sa],C)]]
        neibourSamplesMassDic[seq]=sampleNeibourMass
        aimClass,neibourBindingMassDic[seq]=DSEvidenceCombine(sampleNeibourMass)
        neibourDic={}
        for s in sampleNeibourLabels:
            CTScheme=statsData[s][12]
            if CTScheme in neibourDic:
                neibourDic[CTScheme].append(statsData[s])
            else:
                neibourDic[CTScheme]=[statsData[s]]  
        neibourSamplesLabelDic[seq]=neibourDic
        aimClassDic[seq]=aimClass
    return aimClassDic,neibourSamplesLabelDic,neibourBindingMassDic            
            
            
    