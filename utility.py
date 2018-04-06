# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 13:44:01 2018

@author: akriti
"""

import numpy as np

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
    return Xarr#,conv_out_arr

    
def get_decodedsig(X,motif):
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
        for m in motif:
            m_low=m.lower()
            if m_low in s:
                s=s.replace(m_low, m)      
        LL.append(s) 
    X_val=[]
    for row in LL:
        temp=[]
        i=0
        while i<91:
            val=str(row[i:10+i])
            temp.append(val)
            """
            if val.isupper():
                temp.append(val)
            else:
                val=val.lower()
                temp.append(val)
            """
            i=i+1
        X_val.append(temp)
                    
    return X_val
    
    
def get_convolved_op(X,kernel,bias):
    N,D=X.shape 
    kernel_width=kernel.shape[0]
    kernel_num=kernel.shape[1]
    points=int(D/4)+1-int(kernel_width/4)
    array_X=np.zeros((N,points,kernel_width))
    conv_out=np.zeros((N,points,kernel_num)).astype(np.float32)
    #for arranging X for multiplication
    row_idx=-1    
    for row in X:
        row_idx+=1
        for i in range(points):
            array_X[row_idx,i]=row[(i*4):kernel_width+(i*4)]    
    #Convolution operation
    row_idx=-1
    for row in array_X:
        row_idx+=1
        conv_out[row_idx]=row.dot(kernel)+bias 
    return conv_out