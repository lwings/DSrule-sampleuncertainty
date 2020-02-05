# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

#添加图形属性
plt.xlabel('Therapies')
plt.ylabel('HR')
plt.title('The statistics of HR using case-based method')
a = plt.subplot(1, 1, 1)

plt.ylim=(0, 1)
x2 = [10, 20, 30,40]
x1 = [8, 18, 28,38]
x3 = [12, 22, 32,42]

Y1 = [0.58,0.53,0.59,0.65]
Y2 = [0.8,0.73,0.79,0.77]
Y3 = [0.91,0.93,0.92,0.92]


#这里需要注意在画图的时候加上label在配合plt.legend（）函数就能直接得到图例，简单又方便！

plt.bar(x1, Y1, facecolor='red', width=2, label = 'knn1')
plt.bar(x2, Y2, facecolor='green', width=2, label = 'knn2')
plt.bar(x3, Y3, facecolor='blue', width=2, label = 'knn3')

plt.xticks([10, 20, 30,40], ["CT","RT","ET","TT"])

plt.legend()

plt.show()


#添加图形属性
plt.xlabel('Therapies')
plt.ylabel('HR')
plt.title('The statistics of HR using reason-based method')
a = plt.subplot(1, 1, 1)

plt.ylim=(0, 1)
x2 = [10, 20, 30,40]
x1 = [8, 18, 28,38]


Y1 = [0.66,0.70,0.62,0.79]
Y2 = [0.81,0.75,0.70,0.83]



#这里需要注意在画图的时候加上label在配合plt.legend（）函数就能直接得到图例，简单又方便！

plt.bar(x1, Y1, facecolor='red', width=2, label = 'NCCN')
plt.bar(x2, Y2, facecolor='green', width=2, label = 'RJ')


plt.xticks([9, 19, 29,39], ["CT","RT","ET","TT"])

plt.legend()

plt.show()
