import sys, os
from tqdm import tqdm
import numpy as np
import sys, os
sys.path.append('../')
from torch.utils.data import Dataset
import pandas as pd
from Preprocess.dataCollect import collect_data,set_name
from sklearn.model_selection import train_test_split
from os import path
from gensim.models import KeyedVectors
import pickle
import json
    
    
class Vocab_own():
    def __init__(self,dataframe, model):
        self.itos={}
        self.stoi={}
        self.vocab={}
        self.embeddings=[]
        self.dataframe=dataframe
        self.model=model
    
    ### load embedding given a word and unk if word not in vocab
    ### input: word
    ### output: embedding,word or embedding for unk, unk
    def load_embeddings(self,word):
        try:
            return self.model[word],word
        except KeyError:
            return self.model['unk'],'unk'
    
    ### create vocab,stoi,itos,embedding_matrix
    ### input: **self
    ### output: updates class members
    def create_vocab(self):
        count=1
        for index,row in tqdm(self.dataframe.iterrows(),total=len(self.dataframe)):
            for word in row['Text']:
                vector,word=self.load_embeddings(word)      
                try:
                    self.vocab[word]+=1
                except KeyError:
                    if(word=='unk'):
                        print(word)
                    self.vocab[word]=1
                    self.stoi[word]=count
                    self.itos[count]=word
                    self.embeddings.append(vector)
                    count+=1
        self.vocab['<pad>']=1
        self.stoi['<pad>']=0
        self.itos[0]='<pad>'
        self.embeddings.append(np.zeros((300,), dtype=float))
        self.embeddings=np.array(self.embeddings)
        print(self.embeddings.shape)

    
    
def encodeData(dataframe,vocab,params):
    tuple_new_data=[]
    for index,row in tqdm(dataframe.iterrows(),total=len(dataframe)):
        if(params['bert_tokens']):
            tuple_new_data.append((row['Text'],row['Attention'],row['Label']))
        else:   
            list_token_id=[]
            for word in row['Text']:
                try:
                    index=vocab.stoi[word]
                except KeyError:
                    index=vocab.stoi['unk']
                list_token_id.append(index)
            tuple_new_data.append((list_token_id,row['Attention'],row['Label']))
    return tuple_new_data



def createDatasetSplit(params):
    filename=set_name(params)
    if path.exists(filename):
        ##### REMOVE LATER ######
        #dataset=collect_data(params)
        pass
    else:
        dataset=collect_data(params)
        
    if(path.exists(filename[:-7])):
        with open(filename[:-7]+'/train_data.pickle', 'rb') as f:
            X_train = pickle.load(f)
        with open(filename[:-7]+'/val_data.pickle', 'rb') as f:
            X_val = pickle.load(f)
        with open(filename[:-7]+'/test_data.pickle', 'rb') as f:
            X_test = pickle.load(f)
        if(params['bert_tokens']==False):
            with open(filename[:-7]+'/vocab_own.pickle', 'rb') as f:
                vocab_own=pickle.load(f)
    
        
    else:
        if(params['bert_tokens']==False):
            word2vecmodel1 = KeyedVectors.load("Data/word2vec.model")
            vector = word2vecmodel1['easy']
            assert(len(vector)==300)

        dataset= pd.read_pickle(filename)
        #X_train_dev, X_test= train_test_split(dataset, test_size=0.1, random_state=1,stratify=dataset['Label'])
        #X_train, X_val= train_test_split(X_train_dev, test_size=0.11, random_state=1,stratify=X_train_dev['Label'])
        with open('Data/post_id_divisions.json', 'r') as fp:
            post_id_dict=json.load(fp)
        
        X_train=dataset[dataset['Post_id'].isin(post_id_dict['train'])]
        X_val=dataset[dataset['Post_id'].isin(post_id_dict['val'])]
        X_test=dataset[dataset['Post_id'].isin(post_id_dict['test'])]
        
        if(params['bert_tokens']):
            vocab_own=None    
            vocab_size =0
            padding_idx =0
        else:
            vocab_own=Vocab_own(X_train,word2vecmodel1)
            vocab_own.create_vocab()
            padding_idx=vocab_own.stoi['<pad>']
            vocab_size=len(vocab_own.vocab)

        X_train=encodeData(X_train,vocab_own,params)
        X_val=encodeData(X_val,vocab_own,params)
        X_test=encodeData(X_test,vocab_own,params)
        
        print("total dataset size:", len(X_train)+len(X_val)+len(X_test))

        
        os.mkdir(filename[:-7])
        with open(filename[:-7]+'/train_data.pickle', 'wb') as f:
            pickle.dump(X_train, f)

        with open(filename[:-7]+'/val_data.pickle', 'wb') as f:
            pickle.dump(X_val, f)
        with open(filename[:-7]+'/test_data.pickle', 'wb') as f:
            pickle.dump(X_test, f)
        if(params['bert_tokens']==False):
            with open(filename[:-7]+'/vocab_own.pickle', 'wb') as f:
                pickle.dump(vocab_own, f)
    
    if(params['bert_tokens']==False):
        return X_train,X_val,X_test,vocab_own
    else:
        return X_train,X_val,X_test
              
