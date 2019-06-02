from fun_ware import get_datadic
from import_data import get_data
import tensorflow as tf
import numpy as np
import fun_ware
import knn_ware
from sklearn.preprocessing import normalize

data_dic = get_datadic()
counter={}
tempLabel = [v[-2] for k,v in data_dic.items()]
for i in tempLabel:
    counter[i] = counter.get(i,0) + 1
reflection = {}
idx=0
for k,v in counter.items():
    reflection[k]=idx
    idx+=1
#    
def accuracy(pvalue,tvalue):
    trueCase=0
    for i in range(0,len(tvalue)):
        idx1 = pvalue[i].index(max(pvalue[i]))
        idx2 = tvalue[i].index(1)
        if(idx1==idx2):
            trueCase+=1
    return trueCase*1.0/len(tvalue)

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

statsData,votesData,sideDic,statsWithVotesInfo = fun_ware.getDataWithVoteInfo()

K=10
trainDataLabel=[k for k,v in statsData.items() if k<=2000]
verifyDataLabel=[k for k,v in statsData.items() if k>2000 and k<=2751]
testDataLabel=[k for k,v in statsData.items() if k>2751]
t_f_DataLabel=[k for k,v in statsData.items() if k<=2000]

trainVector = []
trainLabel = []
testVector = []
testLabel = []

for label in testDataLabel:
    sampleNeibourLabels=knn_ware.kNeibours(t_f_DataLabel,label,statsData,K+1)
    section=[]
    for seq in sampleNeibourLabels[1:]:
        section+=(statsData[seq][:-1])
    section+=(statsData[label][:-2])
    testVector.append(section)
    
    temp =[0]*8
    temp[reflection[statsData[label][-2]]] = 1
    testLabel.append(temp)

for label in verifyDataLabel:
#label=verifyDataLabel[0]
    sampleNeibourLabels=knn_ware.kNeibours(trainDataLabel,label,statsData,K)
    section=[]
    for seq in sampleNeibourLabels:
        section+=(statsData[seq][:-1])
    section+=(statsData[label][:-2])
    trainVector.append(section)
    
    temp =[0]*8
    temp[reflection[statsData[label][-2]]] = 1
    trainLabel.append(temp)

x_data = np.array(trainVector)
y_data = np.array(trainLabel)
x_test_data = np.array(testVector)

x_data = normalize(x_data, axis=0, norm='max')
x_test_data = normalize(x_test_data, axis=0, norm='max')

xs = tf.placeholder(tf.float32, [None, 13*(K+1)-1])
ys = tf.placeholder(tf.float32, [None, 8])

#prediction = add_layer(xs, 13*(K+1)-1, 8,  activation_function=tf.nn.softmax)

prediction_1 = add_layer(xs, 13*(K+1)-1, 500,  activation_function=tf.nn.sigmoid)
prediction_2 = add_layer(prediction_1, 500, 500,  activation_function=tf.nn.sigmoid)
#prediction_3= add_layer(prediction_2, 500, 250,  activation_function=tf.nn.sigmoid)
prediction_4= add_layer(prediction_2, 500, 8,  activation_function=tf.nn.softmax)
    
# the error between prediction and real data

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction_4),
                                              reduction_indices=[1]))       # loss
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

sess = tf.Session()
# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)

for i in range(400):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
         print(sess.run(cross_entropy, feed_dict={xs: x_data, ys: y_data}))
        

prediction_value = sess.run(prediction_4, feed_dict={xs: x_test_data})

ac=accuracy(prediction_value.tolist(),testLabel)


