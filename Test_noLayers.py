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
    fpath='Data/Seq_foregrnd_4mer.fa'
    bpath='Data/Seq_bckgrnd_4mer.fa'
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
    epochs = 6
    batch_sz= np.floor(div_Val/10).astype(np.int32)
    n_batches = 10
    print("size is ",batch_sz,"number is ",n_batches)
    #Placeholders
    _X = tf.placeholder(tf.float32, shape=(None, D, 1), name='X')
    _T = tf.placeholder(tf.float32, shape=(None, K), name='T')
    
    
    weight_kernel = (np.random.randn(16,5) / np.sqrt(16+5)).astype(np.float32)
    weight_kernel=weight_kernel.reshape(16,1,5)
    bias_kernel = np.zeros(5, dtype=np.float32)
    
    #conv_output=np.ones((batch_sz,7,5))
    
    conv_kernel = tf.Variable(weight_kernel.astype(np.float32),name='conv1_weight')
    bias_t = tf.Variable(bias_kernel.astype(np.float32),name='conv1_bias')
    #conv_out_alias = tf.Variable(conv_output.astype(np.float32),name='conv1_output')
    
    conv_out = tf.nn.conv1d(_X, conv_kernel, stride = 4, padding = 'VALID', name='conv1')
    conv_out = tf.nn.bias_add(value=conv_out,bias=bias_t)
    
    #tf.assign(conv_out_alias,conv_out)
    
    conv_out_shape = conv_out.get_shape().as_list()
    conv_out_4D = tf.reshape(conv_out, [-1, 1,conv_out_shape[1],conv_out_shape[2]])
    pool_out = tf.nn.max_pool(conv_out_4D, ksize=[1, 1, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
    pool_out= tf.nn.relu(pool_out)
    pool_out_shape = pool_out.get_shape().as_list()    
    flatten_out = tf.reshape(pool_out, [-1, np.prod(pool_out_shape[1:])])
    print(conv_out_shape,"----",flatten_out.get_shape().as_list())
    
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
    
    out_layer = tf.nn.sigmoid(Yish)    
    
    cost = tf.reduce_sum(
                tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=Yish,
                    labels=_T
                )
            )
    
            
    train_op = tf.train.RMSPropOptimizer(0.01).minimize(cost)
    predict_op = tf.argmax(Yish, 1)
    
    correct_pred = tf.equal(tf.argmax(Yish, 1), tf.argmax(_T, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
        
    LLm = []
    LL = []
    acc =[]
    kernel_array=np.zeros((epochs,n_batches,16,5))
    bias_array=np.zeros((epochs,n_batches,5))
    conv_array=np.zeros((epochs,n_batches,batch_sz,7,5))
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)
        for i in range(epochs):
            for j in range(n_batches):
                Xbatch = Xtrain[j*batch_sz:(j*batch_sz + batch_sz),]
                Ybatch = Ytrain_ind[j*batch_sz:(j*batch_sz + batch_sz),]
                t,conv=session.run([train_op,conv_out], feed_dict={_X: Xbatch, _T: Ybatch})
                #print("shape is ",Xbatch.shape,"----",conv.shape)
                kernel = conv_kernel.eval() 
                bias = bias_t.eval()
                kernel_array[i][j]=kernel.reshape(16,5)
                bias_array[i][j]=bias
                conv_array[i][j]=conv
                if j % 4 == 0:
                    test_cost = session.run(cost, feed_dict={_X: Xtest, _T: Ytest_ind})
                    prediction = session.run(predict_op, feed_dict={_X: Xtest})
                    err = error_rate(prediction, Ytest)
                    sig=session.run(out_layer, feed_dict={_X: Xtest})
                    LLm.append(sig)
                    print("Cost / err at iteration i=%d, j=%d: %.4f / %.4f" % (i, j, test_cost, err))
                    LL.append(test_cost)
                    val_acc = session.run(accuracy,feed_dict={_X: Xtest, _T: Ytest_ind})
                    acc.append(val_acc)  
                    
                
    plt.plot(LL)
    plt.show()
    print("Average accuracy",np.mean(np.array(acc)))
    return Xtrain,Ytrain,kernel_array.astype(np.float32),bias_array.astype(np.float32),conv_array.astype(np.float32)

    
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
    """
    b,b_sz,fil_width,fil_num=conv_out.shape
    conv_out=conv_out.reshape(b*b_sz,fil_width,fil_num)
    conv_out_arr=conv_out[[pos], :]   
    Z,N,D,Y=conv_out_arr.shape
    conv_out_arr=conv_out_arr.reshape(N,D,Y)
    """
    return Xarr#,conv_out_arr

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
    
    
X,Y,kernel_full,bias_full,conv_full=getCNNParams()
convution_m=conv_full[-1]
Xarr=extract_motif_cls(X,Y)
actual_X=getdecoded_signal(Xarr)


#Finding kernel and recreating convolution output
kern=kernel_full[-1][8]
bias=bias_full[-1][8]
N,D=Xarr.shape 
#epoch=kernel_array.shape[0]
kernel_width=kern.shape[0]
kernel_num=kern.shape[1]
points=int(D/4)+1-int(kernel_width/4)
array_X=np.zeros((N,points,kernel_width))
conv_out=np.zeros((N,points,kernel_num)).astype(np.float32)

#for arranging X for multiplication
row_idx=-1    
for row in Xarr:
    row_idx+=1
    for i in range(points):
        array_X[row_idx,i]=row[(i*4):kernel_width+(i*4)] 

#Convolution operation
row_idx=-1
for row in array_X:
    row_idx+=1
    conv_out[row_idx]=row.dot(kern)+bias 
    


filter_number,arg_m=finding_motif_pos(conv_out)
filter_op=conv_out[:,:,filter_number]
#arg_maxs=np.argmax(filter_op,axis=1)
pos=np.zeros((filter_op.shape[0],1))
i=0
for row in filter_op:
    val=np.argwhere(row==arg_m)
    if len(val)>1:
        pos[i]=val[0]
    else:
        pos[i]=val
    i+=1

i=0
for row in actual_X:
    val=int(pos[i])
    str=row[val:val+4]
    print(str)
    i+=1


    
    
"""          
i=0
for row in actual_X:
        val=int(arg_maxs[i])
        str=row[val:val+4]
        print(str)
        i+=1                
                
             

def get_motif_positions(conv_out):
    arg_vals=set()
    filter_number=0
    arg_maxs=np.zeros(conv_out.shape[0])
    for z in range(conv_out.shape[2]):
            filter_op=conv_out[:,:,z]
            temp_maxs=np.argmax(filter_op,axis=1)
            temp=set()
            for row in filter_op:
                value=max(row)
                if value>0:
                    temp.add(value)
            if len(arg_vals)<1:
                arg_vals=temp
            else:
                if len(temp)<len(arg_vals):
                    arg_vals=temp
                    filter_number=z
                    arg_maxs=temp_maxs
    return filter_number,arg_maxs
   """             
                
                
                
                
                