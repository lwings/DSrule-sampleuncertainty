from knn_NNAPI import trainNNAndPredict
import fun_ware
import knn_ware
statsData,votesData,sideDic,statsWithVotesInfo = fun_ware.getDataWithVoteInfo()

K=26
trainDataLabel=[k for k in statsWithVotesInfo if k<=3000]
testDataLabel=[k for k in statsWithVotesInfo if k>3000]
neibourSamplesLabelDic,knnAimClassDic= knn_ware.knnClassMajorRule(
    trainDataLabel,testDataLabel,statsData,K)