#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 16:24:18 2019

@author: shenzhengfei
"""
import matplotlib.pyplot as plt


#plot k with prediction precision 
x = range(1,14)
y1=[55.2,55.7,56.3,56.9,57.2,58.4,59.1,60,61,63,66.05,65.67,64.32]
y2=[54.9,55.2,56.8,56.3,57.1,59.4,59.1,62,64,66.04,65,63.67,60.32]
y3=[52.3,52.9,53.3,54.4,55.6,57.2,57.4,59,60.6,64.01,62,60,59]

plt.plot(x,y1,'r',label='experiment 1')
plt.plot(x,y2,'y',label='experiment 2')
plt.plot(x,y3,'b',label='experiment 3')
plt.legend()
plt.xlabel("value of k")
plt.ylabel("value of precision")