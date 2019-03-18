from knn_NNAPI import trainNNAndPredict
import fun_ware
import knn_ware
statsData,votesData,sideDic,statsWithVotesInfo = fun_ware.getDataWithVoteInfo()

K=50
trainDataLabel=[k for k,v in statsData.items() if k<=3000]
testDataLabel=[k for k,v in statsData.items() if k>3000]
#neibourSamplesLabelDic,knnAimClassDic= knn_ware.knnClassMajorRule(
#    trainDataLabel,testDataLabel,statsData,K)

total=0
rightCase=0
for seq in testDataLabel:

    sampleNeibourLabels=knn_ware.kNeibours(trainDataLabel,seq,statsData,K)
    precesion = trainNNAndPredict(sampleNeibourLabels,[seq])
    print(precesion)
    total+=1.0
    if precesion>.5:
        rightCase+=1
        
print(rightCase/total)