import os
import torch
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import getdata
from getdata import Vocab

import cv2
import pickle
import random
class instanceloader(Dataset):

    def __init__(self, root_dir,vocab_dir,batch_size):
        with open(root_dir,'r') as f:
           data=pickle.load(f)
        self.vocab_obj=Vocab(vocab_dir,0,'/home/aniket/pendrive/glove.txt')
        
        length=len(self.data)
        self.actual_data=[]
        self.max_para=0
        self.max_summ=0
        self.NPI=0
        self.NB=0
        self.modelling_data=[]
        for i in data:
            cur_sent,cur_para=data.abstract2sents(i)
            self.modelling_data.append(cur_para)
            self.NPI+=1
            if len(cur_sent.split())>max_summ:
                self.max_summ=len(cur_sent.split())
            if len(cur_para.split())>max_para:
                self.max_para=len(cur_para.split())
            self.actual_data.append([cur_para,cur_sent,1])
            for j in range(10):
                self.NB+=1
                if j<4:
                    r=random.randint(0,length-1)
                    while r==i:
                        r=random.randint(length)
                    r_sent,r_para=data.abstract2sents(r)
                    self.actual_data.append([cur_para,r_sent,0])
                else:
                    r=random.randint(40,50)
                    if r>max_summ:
                        self.max_summ=r
                    r_sent=''
                    for k in range(r):
                        temp_r=random.randint(0,self.vocab_obj.size()-1)
                        r_word=self.vocab_obj.id2word(temp_r)
                        r_sent+=' '+r_word
                    self.actual_data.append([cur_para,r_sent,0])
        self.actual_data_batches=[]
        self.modelling_batches=[]
        self.data_batch_len=int(len(self.actual_data)/batch_size)
        self.modelling_batch_len=int(len(self.modelling_data)/batch_size)
        for i in range(int(len(self.modelling_data)/batch_size)):
            self.modelling_batches.append(self.modelling_data[i*batch_size:i*batch_size+batch_size])
        for i in range(int(len(actual_data)/batch_size)):
            self.actual_data_batches.append(self.actual_data[i*batch_size:i*batch_size+batch_size])
        del data,length,self.modelling_data,self.actual_data
        random.shuffle(self.actual_data)
    
    def get_embed_matrix(self):
        return self.vocab_obj.embed_matrix
    def __getitem__(self, idx,modelling):
        if modelling==0:        
            cur_data=self.actual_data_batches[idx]
            para_index=torch.zeros(len(cur_data),self.max_para)
            summ_index=torch.zeros(len(cur_data),self.max_summ)
            for j,c in enumerate(cur_data):
                para=c[0]
                summ=c[1]
                label=c[2]
                
                para=para.split()
                for i ,p in enumerate(para):
                    para_index[j,i]=self.vocab_obj.word2idx(p)
                summ=summ.split()
                for i,s in enumerate(summ):
                    summ_index[j,i]=self.vocab_obj(s)
            return para_index,summ_index,label
        else:   
            para_index=torch.zeros(len(cur_data),self.max_para)
            #summ_index=torch.zeros(len(cur_data),self.max_summ)
            for j,c in enumerate(cur_data):
                para=c[0]
                
                para=para.split()
                for i ,p in enumerate(para):
                    para_index[j,i]=self.vocab_obj.word2idx(p)
                
            return para_index