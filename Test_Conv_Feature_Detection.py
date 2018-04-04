# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 15:22:40 2018

@author: akriti
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 09:17:37 2018

@author: akriti
"""

from Bio import SeqIO
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.ops.init_ops import ones_initializer

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



Fpath='Foreground_Seq.fa'
Xm,Y=create_data_fastafiles(Fpath,1) 
for x in range(Xm.shape[1]):
    S=Xm[:,x:(x+15)]
    print(S)


"""
Xhat=Xm.reshape(Xm.shape[0],10,4)
N,D=Xm.shape
Xm=Xm.reshape(N,D,1)

X = tf.placeholder(tf.float32, shape=(None, D, 1), name='X')

conv_out = tf.layers.conv1d(inputs=X, filters=2, kernel_size=16, strides=4, padding='valid',
                                use_bias=False,activation = tf.nn.relu,kernel_initializer=tf.ones_initializer)
t=[]
init = tf.global_variables_initializer()
with tf.Session() as session:
        session.run(init)
        for j in range(2):
                    
                    Xbatch = Xm
                    trans = session.run(conv_out,feed_dict={X: Xbatch})
                    t.append(trans)
                    """