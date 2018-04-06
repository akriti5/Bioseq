# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 09:13:17 2018

@author: akriti
"""

from Bio import SeqIO
#from random import randint
import random
import numpy as np

def vis_motifs_content(Xmer_size,path):
    values=set()
    for seq_record in SeqIO.parse(path, "fasta"):
        D=len(seq_record.seq[:])
        i=0
        while i<D:
            st=seq_record.seq[i]
            if st.isupper():
                seque=str(seq_record.seq[i:i+Xmer_size])
                values.add(seque)
                i+=Xmer_size
            else:
                i+=1              
    print(len(values))
    return values

def get_mul_Motifs(motif,num):
    motif_LL=[]
    for i in range(num):
        temp=''.join(random.sample(motif,len(motif)))
        motif_LL.append(temp) 
    return motif_LL


           
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

       
       
def gen_multiMotif_oneMotif(mulMotif_path,path,motif_lst,old_motif):  
     with open(mulMotif_path,"w") as f:
        count=0
        for seq_record in SeqIO.parse(path, "fasta"):
            f.write(">seq"+str(count)+"\n")
            count+=1 
            idx=random.randint(0, 2)
            new_motif=motif_lst[idx]
            value=str(seq_record.seq[:])
            value=value.replace(old_motif, new_motif)
            f.write(value+"\n")
        f.close()
            
def ret_motifsList(motifs):
    LLm=[]
    for i in range(len(motifs)):
        st=motifs.pop()
        LLm.append(st)
    return LLm
                
def gen_randomize_motifs_bgd_seq(path,back_path):
     with open(back_path,"w") as f:
        count=0
        for seq_record in SeqIO.parse(path, "fasta"):
            seq_record=seq_record.lower()
            f.write(">seq"+str(count)+"\n")
            count+=1
            value=str(seq_record.seq[:])
            value=''.join(random.sample(value,len(value)))
            f.write(value+"\n")
        f.close()
    
    
    
path="Data/dna_500_seq.fa"    
mulMotif_path="Data/DNA_seq_mulMotif.fa"
back_path="Data/dna_500_backseq.fa"
Xmer_size=12
motifs=vis_motifs_content(Xmer_size,path)
gen_randomize_motifs_bgd_seq(path,back_path)
#LLm=ret_motifsList(motifs)

    


#motif_lst=get_mul_Motifs(motif,3)
#gen_multiMotif_oneMotif(mulMotif_path,path,motif_lst,motif)
#gen_fasta_backgrnd(mulMotif_path,back_path)


#gen_multiMotif_oneMotif(mulMotif_path,path,motif_lst,motif)
#gen_fasta_backgrnd(mulMotif_path,back_path)
#gen_fasta_backgrnd(path,back_path)
