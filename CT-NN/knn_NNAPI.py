from fun_ware import get_datadic
from import_data import get_data
import tensorflow as tf
import numpy as np

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
    # Make up some real data



#trainVector = []
#trainLabel = []
#testSeq = []
#testVector = []
#testLabel = []
#
#counter={}
#tempLabel = [v[-2] for k,v in data_dic.items()]
#for i in tempLabel:
#    counter[i] = counter.get(i,0) + 1
#reflection = {}
#idx=0
#for k,v in counter.items():
#    reflection[k]=idx
#    idx+=1
#
#for k,v in data_dic.items() :
#    temp =[0]*8
#    temp[reflection[v[-2]]] = 1
#    if k<3000:
#        trainVector.append(v[0:12])
#        trainLabel.append(temp)
#    else:
#        testVector.append(v[0:12])
#        testLabel.append(temp)
#        testSeq.append(k)




def trainNNAndPredict(trainVectorSeq,testVectorSeq):
    trainVector = []
    trainLabel = []
    testVector = []
    testLabel = []
    
    for k in trainVectorSeq:
        trainVector.append(data_dic[k][0:-2])
        temp =[0]*8
        temp[reflection[data_dic[k][-2]]] = 1
        trainLabel.append(temp)
    for k in testVectorSeq:
        testVector.append(data_dic[k][0:-2])
        temp =[0]*8
        temp[reflection[data_dic[k][-2]]] = 1
        testLabel.append(temp)
        
    
    x_data = np.array(trainVector)
    y_data = np.array(trainLabel)
    x_test_data = np.array(testVector)
    # define placeholder for inputs to network
    xs = tf.placeholder(tf.float32, [None, 12])
    ys = tf.placeholder(tf.float32, [None, 8])
    # add output layer
    prediction = add_layer(xs, 12, 8,  activation_function=tf.nn.softmax)
    
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
    
    for i in range(1000):
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        if i % 50 == 0:
            prediction_value = sess.run(prediction, feed_dict={xs: x_data})
#            print(accuracy(prediction_value.tolist(),trainLabel))
    
    prediction_value = sess.run(prediction, feed_dict={xs: x_test_data})
    
#    print("ultimate rate")
#    print(accuracy(prediction_value.tolist(),testLabel))
    return accuracy(prediction_value.tolist(),testLabel)















