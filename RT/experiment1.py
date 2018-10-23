#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import fun_ware
#import import_data
import knn_ware
statsData,votesData,sideDic,statsWithVotesInfo = fun_ware.getDataWithVoteInfo()
#one side case
#statsData: 0-11 digits_____label 
#           12 digit________class
#           13 digit________side
#votesData: 0____side
#            1____user_id
#            2____firstvote
#            3____ultimavote
K=26
trainDataLabel=[k for k in statsWithVotesInfo if k<=3000]
testDataLabel=[k for k in statsWithVotesInfo if k>3000] 

aimClassDic,neibourSamplesLabelDic,neibourBindingMassDic = knn_ware.knnClassWithDSRule(
        trainDataLabel,testDataLabel,statsData,K)

test=sorted([v[-1][1]-v[-2][1] for k,v in neibourBindingMassDic.items()
             if len(v)>=2])