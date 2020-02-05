x=range(1456)
y=[i-1 for i in x]

sum=0
y1=[]
for i in x:
    y1.append(sum)
    sum+=i
    
plt.xlabel('Patient sequential number')
#plt.ylabel('HR')
plt.plot(x,y,label="")
plt.show()
plt.xlabel('Patient sequential number')
plt.plot(x,y1,label="")
plt.show()
