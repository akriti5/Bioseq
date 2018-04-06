# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 13:01:25 2018

@author: akriti
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 11:14:52 2018

@author: akriti
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from Bio import SeqIO
from utility import extract_motif_cls,get_decodedsig,get_convolved_op


def create_data_fastafiles(Fpath,value):
    nA=np.array([1,0,0,0]).astype(np.float32)
    nT=np.array([0,1,0,0]).astype(np.float32)
    nC=np.array([0,0,1,0]).astype(np.float32)
    nG=np.array([0,0,0,1]).astype(np.float32)
    
    records = list(SeqIO.parse(Fpath, "fasta"))
    N=len(records)
    D = [len(rec) for rec in records][0]
    X=np.zeros((N,D*4)).astype(np.float32)
    if value==1:
        Y=np.ones(N).astype(np.float32)
    else:
        Y=np.zeros(N).astype(np.float32)
    i=0
    for seq_record in SeqIO.parse(Fpath, "fasta"):
        seq_record=seq_record.upper()
        arr=np.zeros((D,4))
        for j in range(D):
            if seq_record.seq[j]=="A":
                    arr[j,]=nA
            elif seq_record.seq[j]=="T":
                    arr[j,]=nT
            elif seq_record.seq[j]=="C":
                    arr[j,]=nC
            elif seq_record.seq[j]=="G":
                    arr[j,]=nG
        arr=arr.reshape(D*4)
        X[i,]=arr
        i+=1
    return X,Y


def get_indicator(Y):
    Y=Y.astype(np.int32)
    N=len(Y)
    K=len(set(Y))
    T=np.zeros((N,K))
    for i in range(N):
        T[i,Y[i]]=1
    return T.astype(np.float32)


def error_rate(p, t):
    return np.mean(p != t)


def getCNNParams(): 
    fpath='Data/DNA_seq_mulMotif.fa'
    bpath='Data/DNA_backseq_mulMotif.fa'
    Xf,Yf=create_data_fastafiles(fpath,1)
    Xb,Yb=create_data_fastafiles(bpath,0)

    X=np.concatenate((Xf, Xb), axis=0)
    Y=np.concatenate((Yf, Yb), axis=0)
    N,D=X.shape
    K=len(set(Y))
    X=X.reshape(N,D,1)
    
    X,Y=shuffle(X,Y)
        
    div_Val=np.floor(.6*X.shape[0]).astype(np.int32)    
    Xtrain=X[:div_Val]
    Ytrain=Y[:div_Val]
    Xtest=X[div_Val:]
    Ytest=Y[div_Val:]
    
    
    Ytrain_ind = get_indicator(Ytrain)
    Ytest_ind = get_indicator(Ytest)
        
    print("X size",Xtrain.shape)
    print("Y size",Ytrain.shape)
    
    
   #Parameters
    epochs = 10
    batch_sz = np.floor(div_Val/12).astype(np.int32)
    n_batches = 12#N // batch_sz
    print("size is ",batch_sz,"number is ",n_batches)
    
    #Placeholders
    _X = tf.placeholder(tf.float32, shape=(None, D, 1), name='X')
    _T = tf.placeholder(tf.float32, shape=(None, K), name='T')
    
    
    weight_kernel = (np.random.randn(40,5) / np.sqrt(40+5)).astype(np.float32)
    weight_kernel=weight_kernel.reshape(40,1,5)
    bias_kernel = np.zeros(5, dtype=np.float32)
    
    conv_kernel = tf.Variable(weight_kernel.astype(np.float32),name='conv1_weight')
    bias_t = tf.Variable(bias_kernel.astype(np.float32),name='conv1_bias')
    
    
    
    conv_out = tf.nn.conv1d(_X, conv_kernel, stride = 4, padding = 'VALID')
    conv_out = tf.nn.bias_add(value=conv_out,bias=bias_t)
    
    conv_out_shape = conv_out.get_shape().as_list()
    conv_out_4D = tf.reshape(conv_out, [-1, 1,conv_out_shape[1],conv_out_shape[2]])
    pool_out = tf.nn.max_pool(conv_out_4D, ksize=[1, 1, 9, 1], strides=[1, 1, 2, 1], padding='SAME')
    
    pool_out_shape = pool_out.get_shape().as_list()    
    print("first pooling",pool_out_shape)       
    
    pool_out=tf.reshape(pool_out, [-1,pool_out_shape[2],pool_out_shape[3]])
    
    
    weight_kernel_1 = (np.random.randn(5,5,10) / np.sqrt(5+5+10)).astype(np.float32)
    
    bias_kernel_1 = np.zeros(10, dtype=np.float32)
    
    conv_kernel_1 = tf.Variable(weight_kernel_1.astype(np.float32),name='conv2_weight')
    bias_t_1 = tf.Variable(bias_kernel_1.astype(np.float32),name='conv2_bias')
    
    conv_out_1 = tf.nn.conv1d(pool_out, conv_kernel_1, stride = 2, padding = 'VALID')
    conv_out_1 = tf.nn.bias_add(value=conv_out_1,bias=bias_t_1)
    
    conv_out_1_shape = conv_out_1.get_shape().as_list()

    conv_out_1_4D = tf.reshape(conv_out_1, [-1, 1,conv_out_1_shape[1],conv_out_1_shape[2]])
    pool_out_1 = tf.nn.max_pool(conv_out_1_4D, ksize=[1, 1, 10, 1], strides=[1, 1, 2, 1], padding='SAME')
  
   
    pool_out_1=tf.nn.relu(pool_out_1)
    
    pool_out_1_shape = pool_out_1.get_shape().as_list()   
    
    flatten_out = tf.reshape(pool_out_1, [-1, np.prod(pool_out_1_shape[1:])]) 
        
    print(conv_out_1_shape,"----",flatten_out.get_shape().as_list())
        
    
    M1=90
    W1_init = np.random.randn(np.prod(pool_out_1_shape[1:]), M1) / np.sqrt(np.prod(pool_out_1_shape[1:])+ M1)
    b1_init = np.zeros(M1, dtype=np.float32)
    W2_init = np.random.randn(M1, K) / np.sqrt(M1 + K)
    b2_init = np.zeros(K, dtype=np.float32)
    
    #Placeholders
    W1 = tf.Variable(W1_init.astype(np.float32),name='ANN_lay1_weight')
    b1 = tf.Variable(b1_init.astype(np.float32),name='ANN_lay1_bias')
    W2 = tf.Variable(W2_init.astype(np.float32),name='ANN_lay2_weight')
    b2 = tf.Variable(b2_init.astype(np.float32),name='ANN_lay2_bias')
    
       
    Z1 = tf.nn.relu( tf.matmul(flatten_out, W1) + b1 )
    Yish = tf.matmul(Z1, W2) + b2
    
    out_layer = tf.nn.sigmoid(Yish)    
    
    cost = tf.reduce_sum(
                tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=Yish,
                    labels=_T
                )
            )
    
            
    train_op = tf.train.RMSPropOptimizer(0.001, decay=0.999, momentum=0.9).minimize(cost)
    predict_op = tf.argmax(out_layer, 1)
    
    correct_pred = tf.equal(tf.argmax(Yish, 1), tf.argmax(_T, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
        
    LL = []
    acc =[]
    kernel_array=np.zeros((epochs,n_batches,40,5))
    bias_array=np.zeros((epochs,n_batches,5))
    conv_array=np.zeros((epochs,n_batches,batch_sz,91,5))
    
    
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)
        for i in range(epochs):
            for j in range(n_batches):
                Xbatch = Xtrain[j*batch_sz:(j*batch_sz + batch_sz),]
                Ybatch = Ytrain_ind[j*batch_sz:(j*batch_sz + batch_sz),]
                t,conv=session.run([train_op,conv_out], feed_dict={_X: Xbatch, _T: Ybatch})
                
                kernel = conv_kernel.eval() 
                bias = bias_t.eval()
                kernel_array[i][j]=kernel.reshape(40,5)
                bias_array[i][j]=bias
                conv_array[i][j]=conv
                if j % 4 == 0:
                    test_cost = session.run(cost, feed_dict={_X: Xtest, _T: Ytest_ind})
                    prediction = session.run(predict_op, feed_dict={_X: Xtest})
                    err = error_rate(prediction, Ytest)
                    print("Cost / err at iteration i=%d, j=%d: %.4f / %.4f" % (i, j, test_cost, err))
                    LL.append(test_cost)
                    val_acc = session.run(accuracy,feed_dict={_X: Xtest, _T: Ytest_ind})
                    acc.append(val_acc)  
           
    plt.plot(LL)
    plt.show()
    print("Average accuracy",np.mean(np.array(acc)))
    return Xtest,Ytest,kernel_array.astype(np.float32),bias_array.astype(np.float32),conv_array.astype(np.float32)

     
X,Y,kernel_full,bias_full,conv_full=getCNNParams()
Xarr=extract_motif_cls(X,Y)
motif=['GAGGGACGGG', 'GCAGGGGGGA', 'GGGGGCAGAG']
actual_X=get_decodedsig(Xarr,motif)
"""
conv_out=get_convolved_op(Xarr,kernel_full[8][4],bias_full[8][4])

filter_num=0  
value_set=set()
for z in range(conv_out.shape[2]):
    filter_op=conv_out[:,:,z]
    temp=set()
    for row in filter_op:
        val=np.max(row)
        temp.add(val)
    if len(value_set)<1:
        value_set=temp
        filter_num=z
    elif len(temp)<len(value_set):
        value_set=temp
        filter_num=z
filter_op=conv_out[:,:,filter_num]     
max_values=np.max(filter_op, axis=1)
unique, counts = np.unique(max_values, return_counts=True)
dictionary=dict(zip(unique, counts))
threshold=max(list(dictionary.values()))
threshold=threshold/2
keys=[]
for key, value in dictionary.items():
    if value>threshold:
        keys.append(key)

testing=set()
for key in keys:
    row, col = np.where(filter_op == key)
    for i in range(len(row)):
        val=actual_X[row[i]][col[i]]
        testing.add(val)

print(testing)
"""