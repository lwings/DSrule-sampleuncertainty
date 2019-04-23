import fun_ware
import knn_ware

statsData,votesData,sideDic,statsWithVotesInfo=fun_ware.getDataWithVoteInfo()
firstEntropyDic,UltimaEntropyDic = fun_ware.voteEntropyComp(votesData)