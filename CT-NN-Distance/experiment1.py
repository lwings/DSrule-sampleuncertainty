#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
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
K=5
trainDataLabel=[k for k in statsWithVotesInfo if k<=3000]
testDataLabel=[k for k in statsWithVotesInfo if k>3000] 

aimClassDic,neibourSamplesLabelDic,neibourBindingMassDic = knn_ware.knnClassWithDSRule(
        trainDataLabel,testDataLabel,statsData,K)





def add_layer(inputs, in_size, out_size, activation_function=None):
# add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
            outputs = Wx_plus_b
    else:
            outputs = activation_function(Wx_plus_b)
    return outputs


def accuracy(pvalue,tvalue):
    trueCase=0
    for i in range(0,len(tvalue)):
        idx1 = pvalue[i].index(max(pvalue[i]))
        idx2 = tvalue[i].index(1)
        if(idx1==idx2):
            trueCase+=1
    return trueCase*1.0/len(tvalue)
#compute the reflection
counter={}
tempLabel = [v[-2] for k,v in statsData.items()]
for i in tempLabel:
    counter[i] = counter.get(i,0) + 1
reflection = {}
idx=0
for k,v in counter.items():
    reflection[k]=idx
    idx+=1

prediction = []

for i in range(len(testDataLabel)):

    value = neibourBindingMassDic[testDataLabel[i]]
    new=[0]*len(reflection)
    for k in value:
        new[reflection[k[0]]]=k[1]
    
    prediction.append(new)

testLabel=[]

for i in testDataLabel:
    v = statsData[i][-2]
    new=[0]*len(reflection)
    new[reflection[v]]=1
    
    testLabel.append(new)   


x_data = np.array(prediction)
y_data = np.array(testLabel)


xs = tf.placeholder(tf.float32, [None, 8])
ys = tf.placeholder(tf.float32, [None, 8])
# add output layer
prediction = add_layer(xs, 8, 8,  activation_function=tf.nn.softmax)

# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))       # loss
train_step = tf.train.GradientDescentOptimizer(0.04).minimize(cross_entropy)

sess = tf.Session()
# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)

for i in range(20000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        prediction_value = sess.run(prediction, feed_dict={xs: x_data})
        print(accuracy(prediction_value.tolist(),testLabel))