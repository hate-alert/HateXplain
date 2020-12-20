import torch
import transformers 
from transformers import *
import glob 
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
import random
from transformers import BertTokenizer
#### common utils
from Models.utils import fix_the_random,format_time,get_gpu,return_params
#### metric utils 
from Models.utils import masked_cross_entropy,softmax,return_params
#### model utils
from Models.utils import save_normal_model,save_bert_model,load_model
from tqdm import tqdm
from TensorDataset.datsetSplitter import createDatasetSplit
from TensorDataset.dataLoader import combine_features
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,recall_score,precision_score
import matplotlib.pyplot as plt
import time
import os
import GPUtil
from sklearn.utils import class_weight
import json
from Models.bertModels import *
from Models.otherModels import *
from sklearn.preprocessing import LabelEncoder
from Preprocess.dataCollect import get_test_data,convert_data,get_annotated_data,transform_dummy_data
from TensorDataset.datsetSplitter import encodeData
from tqdm import tqdm, tqdm_notebook
import pandas as pd
import ast
from torch.nn import LogSoftmax
from lime.lime_text import LimeTextExplainer
import numpy as np
import argparse
import GPUtil

# In[3]:





dict_data_folder={
      '2':{'data_file':'Data/dataset.json','class_label':'Data/classes_two.npy'},
      '3':{'data_file':'Data/dataset.json','class_label':'Data/classes.npy'}
}

model_dict_params={
    'bert':'best_model_json/bestModel_bert_base_uncased_Attn_train_FALSE.json',
    'bert_supervised':'best_model_json/bestModel_bert_base_uncased_Attn_train_TRUE.json',
    'birnn':'best_model_json/bestModel_birnn.json',
    'cnngru':'best_model_json/bestModel_cnn_gru.json',
    'birnn_att':'best_model_json/bestModel_birnnatt.json',
    'birnn_scrat':'best_model_json/bestModel_birnnscrat.json'
    
    
}

def select_model(params,embeddings):
    if(params['bert_tokens']):
        print(params['num_classes'])
        if(params['what_bert']=='weighted'):
            model = SC_weighted_BERT.from_pretrained(
            params['path_files'], # Use the 12-layer BERT model, with an uncased vocab.
            num_labels = params['num_classes'], # The number of output labels
            output_attentions = True, # Whether the model returns attentions weights.
            output_hidden_states = False, # Whether the model returns all hidden-states.
            hidden_dropout_prob=params['dropout_bert'],
            params=params
            )
        else:
            print("Error in bert model name!!!!")
        return model
    else:
        text=params['model_name']
        if(text=="birnn"):
            model=BiRNN(params,embeddings)
        elif(text == "birnnatt"):
            model=BiAtt_RNN(params,embeddings,return_att=True)
        elif(text == "birnnscrat"):
            model=BiAtt_RNN(params,embeddings,return_att=True)
        elif(text == "cnn_gru"):
            model=CNN_GRU(params,embeddings)
        elif(text == "lstm_bad"):
            model=LSTM_bad(params)
        else:
            print("Error in model name!!!!")
        return model

    
def standaloneEval(params, test_data=None,extra_data_path=None, topk=2,use_ext_df=False):
    device = torch.device("cpu")
    embeddings=None
    if(params['bert_tokens']):
        train,val,test=createDatasetSplit(params)
        vocab_own=None    
        vocab_size =0
        padding_idx =0
    else:
        train,val,test,vocab_own=createDatasetSplit(params)
        params['embed_size']=vocab_own.embeddings.shape[1]
        params['vocab_size']=vocab_own.embeddings.shape[0]
        embeddings=vocab_own.embeddings
    if(params['auto_weights']):
        y_test = [ele[2] for ele in test] 
        encoder = LabelEncoder()
        encoder.classes_ = np.load(params['class_names'],allow_pickle=True)
        params['weights']=class_weight.compute_class_weight('balanced',np.unique(y_test),y_test).astype('float32')
    if(extra_data_path!=None):
        params_dash={}
        params_dash['num_classes']=2
        params_dash['data_file']=extra_data_path
        params_dash['class_names']=dict_data_folder[str(params['num_classes'])]['class_label']
        temp_read = get_annotated_data(params_dash)
        with open('Data/post_id_divisions.json', 'r') as fp:
            post_id_dict=json.load(fp)
        temp_read=temp_read[temp_read['post_id'].isin(post_id_dict['test'])]
        test_data=get_test_data(temp_read,params,message='text')
        test_extra=encodeData(test_data,vocab_own,params)
        test_dataloader=combine_features(test_extra,params,is_train=False)
    elif(use_ext_df):
        test_extra=encodeData(test_data,vocab_own,params)
        test_dataloader=combine_features(test_extra,params,is_train=False)
    else:
        test_dataloader=combine_features(test,params,is_train=False)
    
    
    
    model=select_model(params,embeddings)
    if(params['bert_tokens']==False):
        model=load_model(model,params)
        
    model.eval()
    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    # Tracking variables
    if((extra_data_path!=None) or (use_ext_df==True) ):
        post_id_all=list(test_data['Post_id'])
    else:
        post_id_all=list(test['Post_id'])
    
    print("Running eval on test data...")
    t0 = time.time()
    true_labels=[]
    pred_labels=[]
    logits_all=[]
    input_mask_all=[]
    
    # Evaluate data for one epoch
    for step, batch in tqdm(enumerate(test_dataloader),total=len(test_dataloader)):

        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)


        # `batch` contains three pytorch tensors:
        #   [0]: input ids 
        #   [1]: attention vals
        #   [2]: attention mask
        #   [3]: labels 
        b_input_ids = batch[0].to(device)
        b_att_val = batch[1].to(device)
        b_input_mask = batch[2].to(device)
        b_labels = batch[3].to(device)


        # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
        model.zero_grad()        
        outputs = model(b_input_ids,
            attention_vals=b_att_val,
            attention_mask=b_input_mask, 
            labels=None,device=device)
        logits = outputs[0]
        
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.detach().cpu().numpy()
        # Calculate the accuracy for this batch of test sentences.
        # Accumulate the total accuracy.
        pred_labels+=list(np.argmax(logits, axis=1).flatten())
        true_labels+=list(label_ids.flatten())
        logits_all+=list(logits)
        input_mask_all+=list(batch[2].detach().cpu().numpy())
    
    
    logits_all_final=[]
    for logits in logits_all:
        logits_all_final.append(softmax(logits))
    
    
    list_dict=[]
    for post_id,logits,pred,ground_truth in zip(post_id_all,logits_all_final,pred_labels,true_labels):
#         if(ground_truth==1):
#             continue
        temp={}
        encoder = LabelEncoder()
        encoder.classes_ = np.load('Data/classes_two.npy',allow_pickle=True)
        pred_label=encoder.inverse_transform([pred])[0]
        ground_label=encoder.inverse_transform([ground_truth])[0]
        temp["annotation_id"]=post_id
        temp["classification"]=pred_label
        temp["ground_truth"]=ground_label
        temp["classification_scores"]={"non-toxic":logits[0],"toxic":logits[1]}
        list_dict.append(temp)
        
    return list_dict,test_data


def get_final_dict(params,test_data,topk):
    list_dict_org,test_data=standaloneEval(params, extra_data_path=test_data, topk=2)
    return list_dict_org

# In[115]:


# def get_final_dict_with_lime(params,model_name,test_data,topk):
#     list_dict_org,test_data=standaloneEval_with_lime(params,model_name,test_data=test_data, topk=topk)
#     test_data_with_rational=convert_data(test_data,params,list_dict_org,rational_present=True,topk=topk)
#     list_dict_with_rational,_=standaloneEval_with_lime(params,model_name,test_data=test_data_with_rational, topk=topk,rational=True)
#     test_data_without_rational=convert_data(test_data,params,list_dict_org,rational_present=False,topk=topk)
#     list_dict_without_rational,_=standaloneEval_with_lime(params,model_name,test_data=test_data_without_rational, topk=topk,rational=True)
#     final_list_dict=[]
#     for ele1,ele2,ele3 in zip(list_dict_org,list_dict_with_rational,list_dict_without_rational):
#         ele1['sufficiency_classification_scores']=ele2['classification_scores']
#         ele1['comprehensiveness_classification_scores']=ele3['classification_scores']
#         final_list_dict.append(ele1)
#     final_list_dict=list_dict_org
#     return final_list_dict


# In[ ]:



# In[88]:

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, 
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


    
if __name__=='__main__': 
    my_parser = argparse.ArgumentParser(description='Which model to use')

    # Add the arguments
    my_parser.add_argument('model_to_use',
                           metavar='--model_to_use',
                           type=str,
                           help='model to use for evaluation')
    
    my_parser.add_argument('attention_lambda',
                           metavar='--attention_lambda',
                           type=str,
                           help='required to assign the contribution of the atention loss')
    
    
    
    args = my_parser.parse_args()
    model_to_use=args.model_to_use    
    params=return_params(model_dict_params[model_to_use],float(args.attention_lambda),2)
    params['variance']=1
    params['num_classes']=2
    fix_the_random(seed_val = params['random_seed'])
    params['class_names']=dict_data_folder[str(params['num_classes'])]['class_label']
    params['data_file']=dict_data_folder[str(params['num_classes'])]['data_file']
    #test_data=get_test_data(temp_read,params,message='text')
    final_dict=get_final_dict(params, params['data_file'],topk=5)
    path_name=model_dict_params[model_to_use]
    path_name_explanation='explanations_dicts/'+path_name.split('/')[1].split('.')[0]+'_bias.json'
    with open(path_name_explanation, 'w') as fp:
        fp.write('\n'.join(json.dumps(i,cls=NumpyEncoder) for i in final_dict))


# In[ ]:

