# -*- coding: utf-8 -*-
import fun_ware
import knn_ware
import numpy as np
import matplotlib.pyplot as plt

class Node(object):
    def __init__(self,id=0,children=[]):
        self.id=id
        self.children=[]
    def append(self,newNode):
        self.children.append(newNode)
        
statsData,votesData,sideDic,statsWithVotesInfo = fun_ware.getDataWithVoteInfo()

root_id = statsWithVotesInfo[0]
root=Node(id=root_id)
nodeNums=1

def AddNode(k,root,ID,trace=[]):

    neighborList=[root]
    for node in root.children:
        neighborList.append(node)
    
    minNode=root
    minDis=999999
    comTimes=len(neighborList)
    for node in neighborList :
        distance = knn_ware.distanceCompute(statsData[ID],statsData[node.id])
        if distance<minDis:
            minDis=distance
            minNode=node
            
    trace.append(minNode)
    
    if minNode.id==root.id:
        alpha=1
        belta=1
        C=1
        K=5
        traceID=[node.id for node in trace]
        aimClassDicV,neibourSamplesLabelDicV,neibourBindingMassDicV=knn_ware.DSRuleWithVoteInfo(
        traceID,[ID],statsData,votesData,K,alpha,belta,C)
        aimClass=aimClassDicV[ID].keys()[0]
        realClass = statsData[ID][-2]
        
        if aimClass == realClass:
            return aimClass,realClass,trace,comTimes
        else:
            if len(root.children)<k:
                root.children.append(Node(id=ID))
                global nodeNums
                nodeNums+=1
            return aimClass,realClass,trace,comTimes

    aimClass,realClass,trace,times = AddNode(k,minNode,ID,trace)
    return aimClass,realClass,trace,times+comTimes


def InitBFRoots(seqList=[statsWithVotesInfo[0]]):
    root=[]
    for seq in seqList:
        














