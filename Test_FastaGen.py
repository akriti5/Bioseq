# -*- coding: utf-8 -*-
"""
This file is used to create a custom length of sequences with motif to be put in fasta files
The generated fasta files take another fasta files' sequences as base and then at random position insert Motif
Before inserting motif it clips the sequence to specified length via variable Xmer_size
The generated fasta file is also used to generate background sequences
"""

from Bio import SeqIO
from random import randint
import random
import numpy as np

def gen_fasta(path,Xmer_size,value,fore_path):
    with open(fore_path,"w") as f:
        count=0
        for seq_record in SeqIO.parse(path, "fasta"):
            f.write(">seq"+str(count)+"\n")
            count+=1
            seq_record=seq_record.lower()
            D=len(seq_record.seq[:])
            
            pos_seq_record=randint(0, D-Xmer_size-1)
            pos_Xmer=randint(0, Xmer_size-len(value)-1)
            
            sequence=str(seq_record.seq[pos_seq_record:pos_seq_record+Xmer_size])
            rep=sequence[pos_Xmer:len(value)+pos_Xmer]
            sequence=sequence.replace(rep,value,1)                     
            f.write(str(sequence)+"\n")
        f.close()
def gen_fasta_backgrnd(fore_path,back_path):
    countA=0
    countT=0
    countC=0
    countG=0
    total_count= 0
    N=0
    D=0
    with open(back_path,"w") as f:
       for seq_record in SeqIO.parse(fore_path, "fasta"):
            N+=1
            seq_recordU=seq_record.upper()
            D=len(seq_record.seq[:])
            for x in range(D):
                    total_count+=1
                    if seq_recordU.seq[x]=="A":
                        countA+=1
                    elif seq_recordU.seq[x]=="T":                      
                        countT+=1
                    elif seq_recordU.seq[x]=="C":
                        countC+=1                       
                    elif seq_recordU.seq[x]=="G":
                        countG+=1
               
       probA=countA/total_count
       probT=countT/total_count
       probC=countC/total_count
       probG=countG/total_count
       ch=['a','t','c','g']
       LL=np.array([probA,probT,probC,probG])
       LL_val= np.asarray(ch)
       minimum_Prob=np.argmin(LL)
       LL_word=LL_val[minimum_Prob]
       #print(LL_word)
       bckgrnd_arr=np.zeros(N*D).astype(np.str)            
       countA=0
       countT=0
       countC=0
       countG=0
       total_count= len(bckgrnd_arr)
       count=0
       while (count < total_count):
                word=random.choice(ch)
                if word=='a' and countA<(probA*total_count):
                   countA+=1
                   bckgrnd_arr[count]=word
                   count+=1              
                elif word=='t' and countT<(probT*total_count):
                   countT+=1
                   bckgrnd_arr[count]=word
                   count+=1
                elif word=='c' and countC<(probC*total_count):
                   countC+=1
                   bckgrnd_arr[count]=word
                   count+=1
                elif word=='g' and countG<(probC*total_count):
                   countG+=1
                   bckgrnd_arr[count]=word
                   count+=1
                else:
                    bckgrnd_arr[count]=LL_word
                    count+=1
             
       bckgrnd_arr=bckgrnd_arr.reshape((N, D)) 
       p=0
       for val in bckgrnd_arr:
           f.write(">seq"+str(p)+"\n") 
           p+=1
           rowSize=len(val)
           for j in range(0,rowSize):
               f.write(val[j])
           f.write("\n")
       f.close()
       return bckgrnd_arr


    
def checklen(path,Xmer_size): 
    count=0
    for seq_record in SeqIO.parse(path, "fasta"):
        D=len(seq_record.seq[:])
        if D!=Xmer_size:
            print(count)
        count+=1
    
    
    
    
path="Data/Test/DNA_500_500.fa"
fore_path="Data/Test/Seq_foreDNA_500_500.fa"
back_path="Data/Test/Seq_backDNA_500_500.fa"
Xmer_size=10
value="TATC"       

#gen_fasta(path,Xmer_size,value,fore_path)
#bckgrnd_arr=gen_fasta_backgrnd(fore_path,back_path)
#checklen(back_path,Xmer_size)