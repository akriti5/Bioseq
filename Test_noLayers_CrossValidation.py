# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 11:15:06 2018

@author: akriti
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 14:26:46 2018

@author: akriti
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 09:17:37 2018

@author: akriti
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from Bio import SeqIO

import pandas as pd
#from tensorflow.python.ops.init_ops import ones_initializer

def read_csvData(path):
    df = pd.read_csv(path,header=None)
    data=df.as_matrix().astype(np.float32)
    X=data[:,:-1]
    Y=data[:,-1]
    X,Y=shuffle(X,Y)
    return X,Y
    
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
    X,Y=read_csvData('Data/DNA_seq.csv')
    N,D=X.shape
    K=len(set(Y))
    X=X.reshape(N,D,1)
    
    #X,Y=shuffle(X,Y)
        
    """
    div_Val=np.floor(.8*X.shape[0]).astype(np.int32)    
    X_train=X[:div_Val]
    Y_train=Y[:div_Val]
    X_test=X[div_Val:]
    Y_test=Y[div_Val:]
    Ytest_ind=get_indicator(Y_test)
    """      
    N,D,Z=X.shape
    Kfold=15
    Kfold_sz=N//Kfold  
    
    epoch=5
    
     #Placeholders
    _X = tf.placeholder(tf.float32, shape=(None, D, 1), name='X')
    _T = tf.placeholder(tf.float32, shape=(None, K), name='T')
    keep= tf.placeholder(tf.float32, name = 'keep')
    
    weight_kernel = (np.random.randn(40,20) / np.sqrt(40+20)).astype(np.float32)
    weight_kernel=weight_kernel.reshape(40,1,20)
    bias_kernel = np.zeros(20, dtype=np.float32)
    
    conv_kernel = tf.Variable(weight_kernel.astype(np.float32),name='conv1_weight')
    bias_t = tf.Variable(bias_kernel.astype(np.float32),name='conv1_bias')
    
    conv_out = tf.nn.conv1d(_X, conv_kernel, stride = 4, padding = 'VALID')
    conv_out = tf.nn.bias_add(value=conv_out,bias=bias_t)
    
    conv_out_shape = conv_out.get_shape().as_list()
    conv_out_4D = tf.reshape(conv_out, [-1, 1,conv_out_shape[1],conv_out_shape[2]])
    pool_out = tf.nn.max_pool(conv_out_4D, ksize=[1, 1, 9, 1], strides=[1, 1, 1, 1], padding='SAME')
    
    pool_out_shape = pool_out.get_shape().as_list()    
    print("first pooling",pool_out_shape)       
    
    pool_out=tf.reshape(pool_out, [-1,pool_out_shape[2],pool_out_shape[3]])
    
    
    weight_kernel_1 = (np.random.randn(5,20,20) / np.sqrt(5+20+20)).astype(np.float32)
    bias_kernel_1 = np.zeros(20, dtype=np.float32)
    
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
    flatten_out = tf.nn.dropout(flatten_out, keep_prob=keep)    

    
    print(conv_out_1_shape,"----",flatten_out.get_shape().as_list())
        
    
    M1=200
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
    
         
    train_op = tf.train.RMSPropOptimizer(0.0001, decay=0.9999, momentum=0.9).minimize(cost)
    predict_op = tf.argmax(out_layer, 1)
    
    correct_pred = tf.equal(tf.argmax(Yish, 1), tf.argmax(_T, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
        

    
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)    
        X_train,Y_train=shuffle(X,Y)   
        for i in range(3):
            for x in range(Kfold):            
                X_te=X_train[(x*Kfold_sz):(x*Kfold_sz+Kfold_sz),:]
                Y_te=Y_train[(x*Kfold_sz):(x*Kfold_sz+Kfold_sz)]
                
                X_tr=np.concatenate((X_train[:(x*Kfold_sz),:],X_train[(x*Kfold_sz+Kfold_sz):,:]), axis=0)
                Y_tr=np.concatenate((Y_train[:(x*Kfold_sz)],Y_train[(x*Kfold_sz+Kfold_sz):]), axis=0)
                
                Ytr_ind = get_indicator(Y_tr)
                Yte_ind = get_indicator(Y_te)
                session.run(train_op, feed_dict={_X: X_tr, _T: Ytr_ind, keep:.2})
                validation_cost = session.run(cost, feed_dict={_X: X_te, _T: Yte_ind,keep:1.0})
                val_prediction = session.run(predict_op, feed_dict={_X: X_te,keep:1.0})
                val_err = error_rate(val_prediction, Y_te)
                val_acc = session.run(accuracy,feed_dict={_X: X_te, _T: Yte_ind,keep:1.0})
                #if (val_acc*100)>58:
                print("Cost / err at iteration  fold=%d: %.4f / %.4f, %.3f" % ( x, validation_cost, val_err,val_acc))
                """
                if (val_acc*100)>55:
                    test_cost = session.run(cost, feed_dict={_X: X_test, _T: Ytest_ind,keep:1.0})
                    test_prediction = session.run(predict_op, feed_dict={_X: X_test,keep:1.0})
                    test_err = error_rate(test_prediction, Y_test)
                    test_acc = session.run(accuracy,feed_dict={_X: X_test, _T: Ytest_ind,keep:1.0})
                    print("Cost / err at test  fold=%d: %.4f / %.4f, %.3f" % ( x, test_cost, test_err,test_acc))
                """    
                
                   
getCNNParams() 