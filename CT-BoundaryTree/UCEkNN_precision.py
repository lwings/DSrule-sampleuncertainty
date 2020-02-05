import fun_ware
import knn_ware
import minKL_ware
import matplotlib.pyplot as plt
statsData,votesData,sideDic,statsWithVotesInfo = fun_ware.getDataWithVoteInfo()

K=10
y=[]
for k in range(11):
    trainDataLabel=statsWithVotesInfo[0:300+k*100]
    testDataLabel=statsWithVotesInfo[300+k*100:400+k*100]
    
    KLDiDic,aimClassDic=minKL_ware.minKLDivergence(trainDataLabel,
                                                testDataLabel,K,statsData)
    neibourSamplesLabelDic,knnAimClassDic= knn_ware.knnClassMajorRule(
        trainDataLabel,testDataLabel,statsData,K)
    
    alpha=1
    belta=1
    C=1
    
    
    aimClassDicV,neibourSamplesLabelDicV,neibourBindingMassDicV=knn_ware.DSRuleWithVoteInfo(
            trainDataLabel,testDataLabel,statsData,votesData,K,alpha,belta,C)
    
    hits=0
    
    
    for label in testDataLabel:
        trueClass = statsData[label][12]
        
        if trueClass == aimClassDicV[label].keys()[0]:
            hits+=1
    y.append(hits-10)
    y.sort()
    print(k)
    
    
plt.xlabel('Patient sequential number')
plt.ylabel('HR')
plt.plot([400+k*100 for k in range(11)],y,label="")
