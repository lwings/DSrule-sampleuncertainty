from import_data import get_data
from import_data import get_data_D
import knn_ware
import math
def get_datadic():
    data_dic={}
    data=get_data()
    label=[0,1,29,32,36,37,38,39,41,42,45,46]
    res=6 #RT_scheme 

    for i in data:
        vector=i[2].encode('unicode-escape').decode('string_escape').split(',')
        if(len(vector)>40):
            for k in range(len(vector)):
                vector[k]=int(vector[k])
            side=vector[24]%2
            nv=[0 for ii in range(len(label)+1)]
            for j in range(len(label)):
                if(label[j]>24):
                    nv[j]=vector[label[j]+side*25]
                else:
                    nv[j]=vector[label[j]]
            nv[-1]=i[res+side*9]
            
            if nv[-1] is None or nv[-1]==0 or nv[-1]==8:
                continue                                                    
            nv.append(side)
            data_dic[i[0]]=nv

    return data_dic

def get_vote_data():
#votesData:  0____side
#            1____user_id
#            2____firstvote
#            3____ultimavote
    voteData = get_data_D()
    voteDataDic = {}
    for line in voteData:
        if line[4]==3 or line[5]==3:
            continue
        if line[0] in voteDataDic:
            voteDataDic[line[0]].append(list(line[2:6]))
        else:
            voteDataDic[line[0]]=[list(line[2:6])]
            
    return voteDataDic

#def save_data(data,location):
#    j=json.dumps(data)
#    with open(location,"w") as f:
#        json.dump(j,f)
#        
#def get_jsondata(location):
#    f1=open(location,"r")
#    data=json.load(f1)
#    f1.close()
#    data=json.loads(data)
#    data={string.atoi(k):v for k,v in data.items()}
#    return data
    
def  getDataWithVoteInfo():
    statsData =  get_datadic()
    votesData = get_vote_data()
    #[side,user_id,pre_CT_id] right:1 left:0
    
    sideDic = {} # left:0 right:1 both:2
    
    for k,v in votesData.items():
        left=0
        right=0
        for i in v:
            if i[0] == 0:
                left =1
            if i[0] == 1:
                right =1
        sideDic[k] = left+right*2-1
        
    statsWithVotesInfo=[k for k,v in statsData.items() if k in votesData 
                    and (statsData[k][-1] == sideDic[k]) and sideDic[k]!=2]
    
    return statsData,votesData,sideDic,statsWithVotesInfo


def voteEntropyComp(votesData):
#votesData:  0____side
#            1____user_id
#            2____firstvote
#            3____ultimavote
    firstEntropyDic = {}
    UltimaEntropyDic = {}
    for pid,voteInfo in votesData.items():
        firstEntropyDic[pid] = 0.0
        UltimaEntropyDic[pid] =0.0
        firstVoteDic={}
        UltimaVoteDic={}
        for info in voteInfo:
            firstVote = info[2]
            UltimaVote = info[3]
            if firstVote in firstVoteDic:
                firstVoteDic[firstVote] += 1.0
            else:
                firstVoteDic[firstVote] = 1.0
            if UltimaVote in UltimaVoteDic:
                UltimaVoteDic[UltimaVote] += 1.0
            else:
                UltimaVoteDic[UltimaVote] = 1.0
        
        for v in firstVoteDic.values():
            firstEntropyDic[pid] -= ( v/len(voteInfo) )  * math.log( ( v/len(voteInfo) ) ,2)
        for v in UltimaVoteDic.values():
            UltimaEntropyDic[pid] -= ( v/len(voteInfo) )  * math.log( ( v/len(voteInfo) ) ,2)

    return firstEntropyDic,UltimaEntropyDic
    
def voteWeightedEntropyComp(votesData):
#votesData:  0____side
#            1____user_id
#            2____firstvote
#            3____ultimavote
    doctorRes = {}
    for patientID,voteResult in votesData.items():
        therapyVotes={}
        for res in voteResult:
            if res[2] in therapyVotes.keys():
                therapyVotes[res[2]]+=1
            else:
                therapyVotes[res[2]]=1
        therapyVotes=sorted(therapyVotes.items(),key=lambda x:x[1], reverse=True)
        ultimateTherapy = therapyVotes[0][0]
        
        for res in voteResult:
            if res[1] not in doctorRes:
                doctorRes[res[1]]=0
            if res[2] == ultimateTherapy:
                doctorRes[res[1]] +=10001
            else:
                doctorRes[res[1]] +=10000
    weight={doctorID:(res%10000)*1.0/(res/10000) for doctorID,res in doctorRes.items()}
    firstEntropyDic = {}
    UltimaEntropyDic = {}
    for pid,voteInfo in votesData.items():
        firstEntropyDic[pid] = 0.0
        UltimaEntropyDic[pid] =0.0
        firstVoteDic={}
        firstVoteWeight={}
        UltimaVoteDic={}
        UltimaVoteWeight={}
        firstVoteNum={}
        for info in voteInfo:
            firstVote = info[2]
            UltimaVote = info[3]
            doctorID=info[1]            
            if firstVote not in firstVoteWeight.keys():
                firstVoteWeight[firstVote]=0
                firstVoteNum[firstVote]=0
            firstVoteWeight[firstVote] += weight[doctorID]
            firstVoteNum[firstVote]+=1.0
#            for k in firstVoteWeight.keys():
#                firstVoteWeight[k]/=firstVoteNum[k]
            if firstVote in firstVoteDic:
                firstVoteDic[firstVote] += 1.0
            else:
                firstVoteDic[firstVote] = 1.0
            if UltimaVote in UltimaVoteDic:
                UltimaVoteDic[UltimaVote] += 1.0
            else:
                UltimaVoteDic[UltimaVote] = 1.0
        
        for k,v in firstVoteDic.items():
            firstEntropyDic[pid] -= 5*firstVoteWeight[k]/v*( v/len(voteInfo) )  * math.log( ( v/len(voteInfo) ) ,2)
        for v in UltimaVoteDic.values():
            UltimaEntropyDic[pid] -= ( v/len(voteInfo) )  * math.log( ( v/len(voteInfo) ) ,2)

    return firstEntropyDic,UltimaEntropyDic

def getWeightOfVotes(votesData):
#votesData:  0____side
#            1____user_id
#            2____firstvote
#            3____ultimavote
    doctorRes = {}
    result={}
    for patientID,voteResult in votesData.items():
        therapyVotes={}
        for res in voteResult:
            if res[2] in therapyVotes.keys():
                therapyVotes[res[2]]+=1
            else:
                therapyVotes[res[2]]=1
        therapyVotes=sorted(therapyVotes.items(),key=lambda x:x[1], reverse=True)
        ultimateTherapy = therapyVotes[0][0]
        result[patientID]=ultimateTherapy
        for res in voteResult:
            if res[1] not in doctorRes:
                doctorRes[res[1]]=0
            if res[2] == ultimateTherapy:
                doctorRes[res[1]] +=10001
            else:
                doctorRes[res[1]] +=10000
    weight={doctorID:(res%10000)*1.0/(res/10000) for doctorID,res in doctorRes.items()}
    weight = sorted(weight.items(),key=lambda x:x[1], reverse=True)
    return weight,result

def voteLsureEntropyComp(votesData,L):
#votesData:  0____side
#            1____user_id
#            2____firstvote
#            3____ultimavote
    weight,result = getWeightOfVotes(votesData)
    firstEntropyDic = {}
    UltimaEntropyDic = {}
    bestDoctors=[i[0] for i in weight][0:L]
    for pid,voteInfo in votesData.items():
        isLsure=1
        firstEntropyDic[pid] = 0.0
#        UltimaEntropyDic[pid] =0.0
        firstVoteDic={}
#        UltimaVoteDic={}
        for info in voteInfo:
            firstVote = info[2]
            doctorID=info[1]
            if(doctorID in bestDoctors):
                if(firstVote!=result[pid]):
                    isLsure=0
#            UltimaVote = info[3]
            if firstVote in firstVoteDic:
                firstVoteDic[firstVote] += 1.0
            else:
                firstVoteDic[firstVote] = 1.0
#            if UltimaVote in UltimaVoteDic:
#                UltimaVoteDic[UltimaVote] += 1.0
#            else:
#                UltimaVoteDic[UltimaVote] = 1.0
        
        if(isLsure==0):
            for v in firstVoteDic.values():
               firstEntropyDic[pid] -= ( v/len(voteInfo) )  * math.log( ( v/len(voteInfo) ) ,2)
#        for v in UltimaVoteDic.values():
#            UltimaEntropyDic[pid] -= ( v/len(voteInfo) )  * math.log( ( v/len(voteInfo) ) ,2)

    return firstEntropyDic,UltimaEntropyDic

def getLocalWeightOfVotes(localVotesData):
    doctorRes = {}
    result={}
    for patientID,voteResult in localVotesData.items():
        therapyVotes={}
        for res in voteResult:
            if res[2] in therapyVotes.keys():
                therapyVotes[res[2]]+=1
            else:
                therapyVotes[res[2]]=1
        therapyVotes=sorted(therapyVotes.items(),key=lambda x:x[1], reverse=True)
        ultimateTherapy = therapyVotes[0][0]
        result[patientID]=ultimateTherapy
        for res in voteResult:
            if res[1] not in doctorRes:
                doctorRes[res[1]]=0
            if res[2] == ultimateTherapy:
                doctorRes[res[1]] +=10001
            else:
                doctorRes[res[1]] +=10000
    weight={}
    totalRate=0
    totalNum=0
    invalidDoctor=[]
    for doctorID,res in doctorRes.items():
        weight[doctorID]=0
        num =(res%10000)*1.0
        den =res/10000
        if(den>0.5*len(localVotesData)):
            weight[doctorID] = num/(den*1.0)
#            totalRate+=weight[doctorID]
#            totalNum+=1.0
        else:
            invalidDoctor.append(doctorID)
#    aveRate = totalRate / totalNum
#    for doctorID in invalidDoctor:
#        weight[doctorID] = aveRate
    weight = sorted(weight.items(),key=lambda x:x[1], reverse=True)
    return weight,result

def localWeightLsureEntropyComp(localVotesData,L):
    weight,result = getLocalWeightOfVotes(localVotesData)
    firstEntropyDic = {}
    bestDoctors=[i[0] for i in weight][0:L]
    for pid,voteInfo in localVotesData.items():
        isLsure=1
        firstEntropyDic[pid] = 0.0
        firstVoteDic={}
        for info in voteInfo:
            firstVote = info[2]
            doctorID=info[1]
            if(doctorID in bestDoctors):
                if(firstVote!=result[pid]):
                    isLsure=0
            if firstVote in firstVoteDic:
                firstVoteDic[firstVote] += 1.0
            else:
                firstVoteDic[firstVote] = 1.0        
        if(isLsure==0):
            for v in firstVoteDic.values():
               firstEntropyDic[pid] -= ( v/len(voteInfo) )  * math.log( ( v/len(voteInfo) ) ,2)
    return firstEntropyDic

def getLocalWeightOfVotesWithGlobalWeight(localVotesData,globalWeight):
    doctorRes = {}
    result={}
    for patientID,voteResult in localVotesData.items():
        therapyVotes={}
        for res in voteResult:
            if res[2] in therapyVotes.keys():
                therapyVotes[res[2]]+=1
            else:
                therapyVotes[res[2]]=1
        therapyVotes=sorted(therapyVotes.items(),key=lambda x:x[1], reverse=True)
        ultimateTherapy = therapyVotes[0][0]
        result[patientID]=ultimateTherapy
        for res in voteResult:
            if res[1] not in doctorRes:
                doctorRes[res[1]]=0
            if res[2] == ultimateTherapy:
                doctorRes[res[1]] +=10001
            else:
                doctorRes[res[1]] +=10000
    weight={}
#    totalRate=0
#    totalNum=0
    invalidDoctor=[]
    for doctorID,res in doctorRes.items():
        weight[doctorID]=0
        num =(res%10000)*1.0
        den =res/10000
        if(den>0.5*len(localVotesData)):
            weight[doctorID] = num/(den*1.0)
#            totalRate+=weight[doctorID]
#            totalNum+=1.0
        else:
            invalidDoctor.append(doctorID)
#    aveRate = totalRate / totalNum
#    for doctorID in invalidDoctor:
#        weight[doctorID] = aveRate
    globalWeightD={v[0]:v[1] for v in globalWeight}        
    for doctorID in invalidDoctor:        
        weight[doctorID] = globalWeightD[doctorID]
    weight = sorted(weight.items(),key=lambda x:x[1], reverse=True)
    return weight,result

def localWeightLsureEntropyCompWithGlobalWeight(localVotesData,L,globalVotesData):
    globalWeight,globalResult =  getWeightOfVotes(globalVotesData)
    weight,result = getLocalWeightOfVotesWithGlobalWeight(localVotesData,globalWeight)
    firstEntropyDic = {}
    bestDoctors=[i[0] for i in weight][0:L]
    for pid,voteInfo in localVotesData.items():
        isLsure=1
        firstEntropyDic[pid] = 0.0
        firstVoteDic={}
        for info in voteInfo:
            firstVote = info[2]
            doctorID=info[1]
            if(doctorID in bestDoctors):
                if(firstVote!=result[pid]):
                    isLsure=0
            if firstVote in firstVoteDic:
                firstVoteDic[firstVote] += 1.0
            else:
                firstVoteDic[firstVote] = 1.0        
        if(isLsure==0):
            for v in firstVoteDic.values():
               firstEntropyDic[pid] -= ( v/len(voteInfo) )  * math.log( ( v/len(voteInfo) ) ,2)
    return firstEntropyDic
    
def getLocalBayesianWeightOfVotes(localVotesData):
    doctorRes = {}
    result={}
    for patientID,voteResult in localVotesData.items():
        therapyVotes={}
        for res in voteResult:
            if res[2] in therapyVotes.keys():
                therapyVotes[res[2]]+=1
            else:
                therapyVotes[res[2]]=1
        therapyVotes=sorted(therapyVotes.items(),key=lambda x:x[1], reverse=True)
        ultimateTherapy = therapyVotes[0][0]
        result[patientID]=ultimateTherapy
        for res in voteResult:
            if res[1] not in doctorRes:
                doctorRes[res[1]]=0
            if res[2] == ultimateTherapy:
                doctorRes[res[1]] +=10001
            else:
                doctorRes[res[1]] +=10000
    weight={}
    totalRate=0
    totalNum=0
    invalidDoctor=[]
    doctorNum={}
    doctorDen={}
    for doctorID,res in doctorRes.items():
        weight[doctorID]=0
        num =(res%10000)*1.0
        den =res/10000
        doctorNum[doctorID] = num
        doctorDen[doctorID] = den
        if(den>0.5*len(localVotesData)):
            weight[doctorID] = num/(den*1.0)
            totalRate+=weight[doctorID]
            totalNum+=1.0
        else:
            invalidDoctor.append(doctorID)
    aveRate = totalRate / totalNum
    aveNum = aveRate*1.0*len(localVotesData)
    for doctorID in invalidDoctor:
        weight[doctorID] = (aveNum+doctorNum[doctorID])*1.0/(doctorDen[doctorID]+len(localVotesData))
    weight = sorted(weight.items(),key=lambda x:x[1], reverse=True)
    return weight,result

def localBayesianWeightLsureEntropyComp(localVotesData,L):
    weight,result = getLocalBayesianWeightOfVotes(localVotesData)
    firstEntropyDic = {}
    bestDoctors=[i[0] for i in weight][0:L]
    for pid,voteInfo in localVotesData.items():
        isLsure=1
        firstEntropyDic[pid] = 0.0
        firstVoteDic={}
        for info in voteInfo:
            firstVote = info[2]
            doctorID=info[1]
            if(doctorID in bestDoctors):
                if(firstVote!=result[pid]):
                    isLsure=0
            if firstVote in firstVoteDic:
                firstVoteDic[firstVote] += 1.0
            else:
                firstVoteDic[firstVote] = 1.0        
        if(isLsure==0):
            for v in firstVoteDic.values():
               firstEntropyDic[pid] -= ( v/len(voteInfo) )  * math.log( ( v/len(voteInfo) ) ,2)
    return firstEntropyDic    