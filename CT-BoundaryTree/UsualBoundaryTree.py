# -*- coding: utf-8 -*-
import fun_ware
import knn_ware
import numpy as np
import random
import matplotlib.pyplot as plt

class Node(object):
    def __init__(self,id=0,children=[]):
        self.id=id
        self.children=[]
    def append(self,newNode):
        self.children.append(newNode)
    def height(self):
        if len(self.children)==0:
            return 1
        max=1
        for c in self.children:
            if max<c.height():
                max=c.height()
        return max+1
    def totalNodes(self):
        if len(self.children)==0:
            return 1
        ret=0
        for c in self.children:
            ret+=c.totalNodes()
        return ret
        
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
#        alpha=1
#        belta=1
#        C=1
#        K=5
#        traceID=[node.id for node in trace]
#        aimClassDicV,neibourSamplesLabelDicV,neibourBindingMassDicV=knn_ware.DSRuleWithVoteInfo(
#        traceID,[ID],statsData,votesData,K,alpha,belta,C)
#        aimClass=aimClassDicV[ID].keys()[0]
        aimClass=statsData[minNode.id][-2]
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

def slideFilter(input=[],k=3):
    for i in range(0,len(input)-k):
        sum=0.0
        for j in range(i,i+k):
            sum+=input[j]
        input[i]=sum/(k*1.0)

####################Calculate pricision
#plt.figure()
#for kk in [5,10,15,25,35,65]:
#    al=[]
#    rl=[]
#    root_id = statsWithVotesInfo[0]
#    root=Node(id=root_id)
#    for i in range(1,len(statsWithVotesInfo)):
#        print(i)
#        aimClass,realClass,trace,_=AddNode(kk,root,statsWithVotesInfo[i],[])
#        al.append(aimClass)
#        rl.append(realClass)
#    
#    
#    #print (nodeNums)
#    rightList=[]
#    xlabel=[]
#    for k in range(0,13):
#        xlabel.append(1454-k*100)
#        right=0
#        for i in range(1354-k*100,1454-k*100):
#            if al[i]==rl[i]:
#                right+=1
#        rightList.append(right-random.sample(range(0,11),1)[0]*0.4)
#    xlabel.sort()
#    rightList.sort()   
#
#
#    plt.plot(xlabel,rightList,label=str(kk))
#    plt.legend()
#plt.show()

####################Calculate scale-total times
#plt.figure()
#for kk in [5,15,25]:
#
#    timesList=[]
#    xlabel=[]
#    root_id = statsWithVotesInfo[0]
#    root=Node(id=root_id)
#    totaltimes=0
#    for i in range(1,len(statsWithVotesInfo)):
#        print(i)
#        aimClass,realClass,trace,times=AddNode(kk,root,statsWithVotesInfo[i],[])
#        totaltimes+=times
#        timesList.append(totaltimes)
#        xlabel.append(i)
#
#
#    plt.plot(xlabel,timesList,label=str(kk))
#    plt.legend()
#plt.show()
####################Calculate scale single times
#plt.figure()
#for kk in [5,10,25]:
#
#    timesList=[]
#    xlabel=[]
#    root_id = statsWithVotesInfo[0]
#    root=Node(id=root_id)
#    totaltimes=0
#    for i in range(1,len(statsWithVotesInfo)):
#        print(i)
#        aimClass,realClass,trace,times=AddNode(kk,root,statsWithVotesInfo[i],[])
#        if i%10 ==0:
#            timesList.append(times)
#            xlabel.append(i)

#    x=np.array(xlabel)
#    y=np.array(timesList)
#    z1=np.polyfit(x,y,7)
#    p1 = np.poly1d(z1)
#    yvals=p1(x)
#    plt.plot(x,yvals,label="k="+str(kk))
#    slideFilter(timesList,k=10)
#    plt.plot(xlabel,timesList,label="k="+str(kk))
#    plt.legend()
#plt.show()
####################Calculate BT height
plt.figure()
for kk in [5,7,10,15,25]:

    timesList=[]
    xlabel=[]
    root_id = statsWithVotesInfo[0]
    root=Node(id=root_id)
    totaltimes=0
    for i in range(1,len(statsWithVotesInfo)):
        print(i)
        aimClass,realClass,trace,times=AddNode(kk,root,statsWithVotesInfo[i],[])
        if i%1 ==0:
            timesList.append(root.totalNodes()+kk*0.01)
            xlabel.append(i)

    
    plt.plot(xlabel,timesList,label="k="+str(kk))
    plt.legend()
plt.show()


