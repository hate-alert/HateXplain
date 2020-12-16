import time
import datetime
import numpy as np
from sklearn.metrics import f1_score
import random
import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from collections import Counter
import os
from tqdm import tqdm_notebook,tqdm
import pandas as pd 
import torch
import torch.nn as nn


    

def cross_entropy(input1, target, size_average=True):
    """ Cross entropy that accepts soft targets
    Args:
         pred: predictions for neural network
         targets: targets, can be soft
         size_average: if false, sum is returned instead of mean

    Examples::

        input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
        input = torch.autograd.Variable(out, requires_grad=True)

        target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        target = torch.autograd.Variable(y1)
        loss = cross_entropy(input, target)
        loss.backward()
    """
    logsoftmax = nn.LogSoftmax(dim=0)
    return torch.sum(-target * logsoftmax(input1))
    # if size_average:
    #     return torch.mean(torch.sum(-target * logsoftmax(input), dim=1))
    # else:
    #     return torch.sum(torch.sum(-target * logsoftmax(input), dim=1))


def masked_cross_entropy(input1,target,mask):
#     list_count=[]
#     #print("len mask",len(mask))
#     for h in range(len(mask)):
#         count=0
#         #print(x_batch_mask[h])
#         for element in mask[h]:
#             if element==0:
#                 break
#             else:
#                 count+=1
#         list_count.append(count)
    #print(list_count)
    cr_ent=0
    for h in range(0,mask.shape[0]):
        #print(input1.shape)
        cr_ent+=cross_entropy(input1[h][mask[h]],target[h][mask[h]])
    
    return cr_ent/mask.shape[0]



def fix_the_random(seed_val = 42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)



def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)



def flat_fscore(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, pred_flat, average='macro')






def load_model(model, params, use_cuda=False):
    if(params['att_lambda']>=1):
        params['att_lambda']=int(params['att_lambda'])
    model_path='Saved/'+params['model_name']+'_'+params['seq_model']+'_'+str(params['hidden_size'])+'_'+str(params['num_classes'])+'_'+str(params['att_lambda'])+'.pth'    
    """Load model."""
    map_location = 'cpu'
#     if use_cuda and torch.cuda.is_available():
#         map_location = 'cuda'
    model.load_state_dict(torch.load(model_path, map_location))
    return model





class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)


    
def save_normal_model(model, params):
    """Save model."""
    model_path='Saved/'+params['model_name']+'_'+params['seq_model']+'_'+str(params['hidden_size'])+'_'+str(params['num_classes'])+'_'+str(params['att_lambda'])+'.pth'
    torch.save(model.state_dict(), model_path)



    
def save_bert_model(model,tokenizer,params):
        output_dir = 'Saved/'+params['path_files']+'_'
        if(params['train_att']):
            output_dir =  output_dir+ str(params['supervised_layer_pos'])+'_'+str(params['num_supervised_heads'])
        output_dir=output_dir+'_'+str(params['num_classes'])+'_'+str(params['att_lambda'])+'/'
        print(output_dir)
        # Create output directory if needed
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("Saving model to %s" % output_dir)

        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)


# import torch
# import torch.nn as nn


# def save_model(model, model_path):
#     """Save model."""
#     torch.save(model.state_dict(), model_path)


# def load_model(model, model_path, use_cuda=False):
#     """Load model."""
#     map_location = 'cpu'
#     if use_cuda and torch.cuda.is_available():
#         map_location = 'cuda:0'
#     model.load_state_dict(torch.load(model_path, map_location))
#     return model



