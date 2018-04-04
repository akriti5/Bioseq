# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 17:00:07 2018

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
    fpath='Data/Test/Seq_foregrnd_4mer.fa'
    bpath='Data/Test/Seq_bckgrnd_4mer.fa'
    Xf,Yf=create_data_fastafiles(fpath,1)
    Xb,Yb=create_data_fastafiles(bpath,0)
    
    X=np.concatenate((Xf, Xb), axis=0)
    Y=np.concatenate((Yf, Yb), axis=0)
    N,D=X.shape
    K=len(set(Y))
    X=X.reshape(N,D,1)
    del fpath
    del bpath
    del Xf;del Yf;del Xb;del Yb;    
    X,Y=shuffle(X,Y)
            
        
    Y_ind = get_indicator(Y)
        
    epochs = 4
    
        
        #Placeholders
    _X = tf.placeholder(tf.float32, shape=(None, D, 1), name='X')
    _T = tf.placeholder(tf.float32, shape=(None, K), name='T')
        
        
    weight_kernel = (np.random.randn(16,5) / np.sqrt(16+5)).astype(np.float32)
    weight_kernel=weight_kernel.reshape(16,1,5)
    bias_kernel = np.zeros(5, dtype=np.float32)
        
    conv_kernel = tf.Variable(weight_kernel.astype(np.float32),name='conv1_weight')
    bias_t = tf.Variable(bias_kernel.astype(np.float32),name='conv1_bias')
        
    conv_out = tf.nn.conv1d(_X, conv_kernel, stride = 4, padding = 'VALID')
    conv_out = tf.nn.bias_add(value=conv_out,bias=bias_t)
           
    conv_out_shape = conv_out.get_shape().as_list()
    conv_out_4D = tf.reshape(conv_out, [-1, 1,conv_out_shape[1],conv_out_shape[2]])
       
    pool_out = tf.nn.max_pool(conv_out_4D, ksize=[1, 1, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
    pool_out= tf.nn.relu(pool_out)
    pool_out_shape = pool_out.get_shape().as_list()    
        
    flatten_out = tf.reshape(pool_out, [-1, np.prod(pool_out_shape[1:])])
    print(pool_out_shape,"----",flatten_out.get_shape().as_list())
        
    M1=30
    W1_init = np.random.randn(np.prod(pool_out_shape[1:]), M1) / np.sqrt(np.prod(pool_out_shape[1:])+ M1)
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
        
        
        
    cost = tf.reduce_sum(
                    tf.nn.softmax_cross_entropy_with_logits_v2(
                        logits=Yish,
                        labels=_T
                    )
                )
        
                
    train_op = tf.train.RMSPropOptimizer(0.01, decay=0.99).minimize(cost)
    predict_op = tf.argmax(Yish, 1)
        
    correct_pred = tf.equal(tf.argmax(Yish, 1), tf.argmax(_T, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
            
    LL = []
    acc =[]
    kernel_array=np.zeros((epochs,16,5))
    bias_array=np.zeros((epochs,5))
    conv_array=np.zeros((epochs,6,7,5))
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)
        for i in range(epochs):
                t,conv=session.run([train_op,conv_out], feed_dict={_X: X, _T: Y_ind})
                kernel = conv_kernel.eval() 
                bias = bias_t.eval()
                kernel_array[i]=kernel.reshape(16,5)
                bias_array[i]=bias
                conv_array[i]=conv
                test_cost = session.run(cost, feed_dict={_X: X, _T: Y_ind})
                prediction = session.run(predict_op, feed_dict={_X: X})
                err = error_rate(prediction, Y)
                print("Cost / err at iteration i=%d : %.4f / %.4f" % (i, test_cost, err))
                LL.append(test_cost)
                val_acc = session.run(accuracy,feed_dict={_X: X, _T: Y_ind})
                acc.append(val_acc)  
    plt.plot(LL)
    plt.show()            
    print("Average accuracy",np.mean(np.array(acc)))
    return X,Y,kernel_array.astype(np.float32),bias_array.astype(np.float32),conv_array.astype(np.float32)

def getdecoded_signal(X):
    N,D=X.shape
    D=int(D/4)
    X=X.reshape(N,D,4)
    LL=[]
    for row in X:
        max_indx=np.argmax(row, axis=1)
        l = []
        for x in max_indx:
            if x==0:
                l.append('a')
            elif x==1:
                l.append('t')
            elif x==2:
                l.append('c')
            else:
                l.append('g')
        s = ''.join(l)
        LL.append(s)  
    return LL

    

#without epochs              
def construct_conv_out(X,kernel_array,bias_array):
    N,D=X.shape 
    kernel_width=kernel_array.shape[0]#change when epoch use included
    kernel_num=kernel_array.shape[1]
    #X=X.reshape(N,D)
    
    points=int(D/4)+1-int(kernel_width/4)
    array_X=np.zeros((N,points,kernel_width))
    conv_out=np.zeros((N,points,kernel_num))
    row_idx=-1    
    for row in X:
        row_idx+=1
        for i in range(points):
            array_X[row_idx,i]=row[(i*4):kernel_width+(i*4)] 
    row_idx=-1   
    for row in array_X:
        row_idx+=1
        conv_out[row_idx]=row.dot(kernel_array)+bias_array   
    return conv_out



#with epochs
def construct_conv_out_e(X,kernel_array,bias_array):
    N,D=X.shape 
    epoch=kernel_array.shape[0]
    kernel_width=kernel_array.shape[1]
    kernel_num=kernel_array.shape[2]
    
    
    points=int(D/4)+1-int(kernel_width/4)
    array_X=np.zeros((N,points,kernel_width))
    conv_out=np.zeros((epoch,N,points,kernel_num))
    row_idx=-1    
    for row in X:
        row_idx+=1
        for i in range(points):
            array_X[row_idx,i]=row[(i*4):kernel_width+(i*4)] 
    
    for e in range(epoch):
        kern=kernel_array[e]
        bias=bias_array[e]
        row_idx=-1
        for row in array_X:
            row_idx+=1
            conv_out[e,row_idx]=row.dot(kern)+bias 
    return conv_out.astype(np.float32)

def extract_motif_cls(X,Y):
    N,D,Z=X.shape 
    X=X.reshape(N,D)
    pos=[]
    for i in range(len(Y)):
        if Y[i]==1:
            pos.append(i)
    Xarr=X[[pos], :]
    Z,N,D=Xarr.shape
    Xarr=Xarr.reshape(N,D)
    return Xarr

def finding_motif_pos(conv_out):
    argument_1=set()
    argument_2=set()
    arg_vals=set([0])
    filter_num=0
    for z in range(conv_out.shape[2]):
        filter_op=conv_out[:,:,z]
        argument_1.clear()
        argument_2.clear()
        for row in filter_op:
            if len(argument_1)<1:
                argument_1.update(row)
            else:
                 if len(argument_2)<1:
                     argument_2.update(row)
                     temp=set.intersection(argument_1, argument_2)   
                     
                 else:
                     argument_2.clear()
                     argument_2.update(row)
                     temp=set.intersection(argument_2, temp)
                     for x in temp:
                         if x>0 and x>max(arg_vals):
                             arg_vals.clear()
                             arg_vals.add(x)
                             filter_num=z
    print("Values of arguments ",arg_vals)    
    return filter_num,max(arg_vals)
    
def printing_motif(filter_op,val):
    pos=np.zeros((filter_op.shape[0],1))
    i=0
    for row in filter_op:
        pos[i]=np.argwhere(row==val)
        i+=1
    i=0
    for row in actual_X:
        val=int(pos[i])
        str=row[val:val+4]
        print(str)
        i+=1
    
X,Y,kernel_array,bias_array,conv_array=getCNNParams()
Xarr=extract_motif_cls(X,Y)
actual_X=getdecoded_signal(Xarr)
conv_out=construct_conv_out_e(Xarr,kernel_array,bias_array)
conv_out_ep=conv_out[3]

filter_num,val=finding_motif_pos(conv_out_ep)

filter_op=conv_out_ep[:,:,filter_num] 
printing_motif(filter_op,val)
