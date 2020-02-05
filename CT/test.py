# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from fun_ware import get_vote_data

data=get_vote_data()
r={}
vtimes={}
x=[]
y=[]
z=[]
for k,vlist in data.items():
    for v in vlist:
        if v[1] not in r:
            r[v[1]]=0
            vtimes[v[1]]=0

for k,vlist in data.items():
    firstVoteList=[v[2] for v in vlist]
    uv=max(firstVoteList, key=firstVoteList.count)
    
    for v in vlist:
        vtimes[v[1]]+=1
        if v[2] == uv:
            r[v[1]]+=1

    x.append(r[27])
    y.append(r[28])
    z.append(r[31])
rate={}
for k,v in vtimes.items():
    rate[k]=(r[k]*1.0/vtimes[k])
    
#plt.xlabel('Doctor ID')
#plt.ylabel('success rate')
#plt.title('The statistics of success rate')
#
#plt.bar(rate.keys(), rate.values(), facecolor='blue', width=0.4)
#plt.show()
#
#
#plt.xlabel('amount of patient')
#plt.ylabel('accmulate success times')
#plt.title('The statistics of success times')
#plt.plot(range(len(x)),x,'r',lw=1.5)
#plt.plot(range(len(x)),y,'b',lw=1.5)
#plt.plot(range(len(x)),z,'g',lw=1.5)
#plt.show()


values=rate.values()
values.sort(reverse=True)
plt.axis([-1,15.9,0,1.1])
plt.xlabel('sequential number')
plt.ylabel('success rate')
plt.title('The statistics of success rate')

plt.bar(range(len(values)), values, facecolor='blue', width=0.4)
