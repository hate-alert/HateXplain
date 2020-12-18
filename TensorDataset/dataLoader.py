import torch
import transformers
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
from sklearn.preprocessing import LabelEncoder



def custom_att_masks(input_ids):
    attention_masks = []

    # For each sentence...
    for sent in input_ids:

        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in sent]

        # Store the attention mask for this sentence.
        attention_masks.append(att_mask)
    return attention_masks

def combine_features(tuple_data,params,is_train=False):
    input_ids =  [ele[0] for ele in tuple_data]
    att_vals = [ele[1] for ele in tuple_data]
    labels = [ele [2] for ele in tuple_data]
    
   
    encoder = LabelEncoder()
    
    encoder.classes_ = np.load(params['class_names'],allow_pickle=True)
    labels=encoder.transform(labels)
    
    input_ids = pad_sequences(input_ids,maxlen=int(params['max_length']), dtype="long", 
                          value=0, truncating="post", padding="post")
    att_vals = pad_sequences(att_vals,maxlen=int(params['max_length']), dtype="float", 
                          value=0.0, truncating="post", padding="post")
    att_masks=custom_att_masks(input_ids)
    dataloader=return_dataloader(input_ids,labels,att_vals,att_masks,params,is_train)
    return dataloader

def return_dataloader(input_ids,labels,att_vals,att_masks,params,is_train=False):
    inputs = torch.tensor(input_ids)
    labels = torch.tensor(labels,dtype=torch.long)
    masks = torch.tensor(np.array(att_masks),dtype=torch.uint8)
    attention = torch.tensor(np.array(att_vals),dtype=torch.float)
    data = TensorDataset(inputs,attention,masks,labels)
    if(is_train==False):
        sampler = SequentialSampler(data)
    else:
        sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=params['batch_size'])
    return dataloader

