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


class modelPred():
    def __init__(self,model_to_use,params):
        self.params=params
#         self.params["device"]='cuda'
        self.embeddings=None
        if(self.params['bert_tokens']):
            self.train,self.val,self.test=createDatasetSplit(params)
            self.vocab=None    
            vocab_size =0
            padding_idx =0
        else:
            self.train,self.val,self.test,vocab_own=createDatasetSplit(params)
            self.params['embed_size']=vocab_own.embeddings.shape[1]
            self.params['vocab_size']=vocab_own.embeddings.shape[0]
            self.vocab=vocab_own
            self.embeddings=vocab_own.embeddings
        if torch.cuda.is_available() and self.params['device']=='cuda':    
            # Tell PyTorch to use the GPU.    
            self.device = torch.device("cuda")
            deviceID = get_gpu(self.params)
            torch.cuda.set_device(deviceID[0])
        else:
            print('Since you dont want to use GPU, using the CPU instead.')
            self.device = torch.device("cpu")

        
        
        self.model=select_model(self.params,self.embeddings)
        if(self.params['bert_tokens']==False):
            #pass
            self.model=load_model(self.model,self.params)
        if(self.params["device"]=='cuda'):
            self.model.cuda()
        self.model.eval()
    
    def return_probab(self,sentences_list):
        """Input: should be a list of sentences"""
        """Ouput: probablity values"""
        params=self.params
        device = self.device
        
        if(params['auto_weights']):
            y_test = [ele[2] for ele in self.test] 
            encoder = LabelEncoder()
            encoder.classes_ = np.load('Data/classes.npy')
            params['weights']=class_weight.compute_class_weight('balanced',np.unique(y_test),y_test).astype('float32')


        temp_read=transform_dummy_data(sentences_list)
        test_data=get_test_data(temp_read,params,message='text')
        test_extra=encodeData(test_data,self.vocab,params)
        test_dataloader=combine_features(test_extra,params,is_train=False)


        
        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        # Tracking variables
        post_id_all=list(test_data['Post_id'])

        print("Running eval on test data...")
        t0 = time.time()
        true_labels=[]
        pred_labels=[]
        logits_all=[]
        #attention_all=[]
        input_mask_all=[]

        # Evaluate data for one epoch
        for step,batch in enumerate(test_dataloader):

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
            #model.zero_grad()        
            outputs = self.model(b_input_ids,
                attention_vals=b_att_val,
                attention_mask=b_input_mask, 
                labels=None,device=device)
            logits = outputs[0]
            #print(logits)
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.detach().cpu().numpy()

    

            # Calculate the accuracy for this batch of test sentences.
            # Accumulate the total accuracy.
            pred_labels+=list(np.argmax(logits, axis=1).flatten())
            true_labels+=list(label_ids.flatten())
            logits_all+=list(logits)
            #attention_all+=list(attention_vectors)
            input_mask_all+=list(batch[2].detach().cpu().numpy())


        logits_all_final=[]
        for logits in logits_all:
            logits_all_final.append(list(softmax(logits)))

        return np.array(logits_all_final)


# In[ ]:



def standaloneEval_with_lime(params, model_to_use,test_data=None,topk=2,rational=False):
    encoder = LabelEncoder()
    encoder.classes_ = np.load('Data/classes.npy')
    explainer = LimeTextExplainer(class_names=list(encoder.classes_),split_expression='\s+',random_state=333,bow=False)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=False)
    
    list_dict=[]
    
    modelClass=modelPred(model_to_use,params)
    
    
    
    if(rational==True):
        sentence_list=[]
        post_id_list=[]
        for index,row in tqdm(test_data.iterrows(),total=len(test_data)):
            #print(row)
            if(row['Label']=='normal'):
                continue
            if(params['bert_tokens']):
                tokens=tokenizer.convert_ids_to_tokens(row['Text'])[1:-1]
                sentence=tokenizer.convert_tokens_to_string(tokens)
            else:
                tokens=row['Text']
                sentence= " ".join(tokens)
            sentence_list.append(sentence)
            post_id_list.append(row['Post_id'])
        
        probab_list=modelClass.return_probab(sentence_list)
        for post_id,proba in zip(post_id_list,list(probab_list)):
            temp={}
            temp["annotation_id"]=post_id
            temp["classification_scores"]={"hatespeech":proba[0],"normal":proba[1],"offensive":proba[2]}
            list_dict.append(temp)
    
    
    else:
        
        for index,row in tqdm(test_data.iterrows(),total=len(test_data)):
            if(row['Label']=='normal'):
                continue
            if(params['bert_tokens']):
                tokens=tokenizer.convert_ids_to_tokens(row['Text'])[1:-1]
                sentence=tokenizer.convert_tokens_to_string(tokens)
            else:
                tokens=row['Text']
                sentence= " ".join(tokens)
                
            temp={}

            exp = explainer.explain_instance(sentence, modelClass.return_probab, num_features=6, top_labels=3,num_samples=params['num_samples'])
            pred_id=np.argmax(exp.predict_proba)
            pred_label=encoder.inverse_transform([pred_id])[0]
            ground_label=row['Label']
            temp["annotation_id"]=row['Post_id']
            temp["classification"]=pred_label
            temp["classification_scores"]={"hatespeech":exp.predict_proba[0],"normal":exp.predict_proba[1],"offensive":exp.predict_proba[2]}

            attention = [0]*len(sentence.split(" "))

            explanation = exp.as_map()[pred_id]
            for exp in explanation:
                if(exp[1]>0):
                    attention[exp[0]]=exp[1]

            if(params['bert_tokens']==True):
                final_explanation=[0]
                tokens=sentence.split(" ")
                for i in range(len(tokens)):
                    temp_tokens=tokenizer.encode(tokens[i],add_special_tokens = False)
                    for j in range(len(temp_tokens)):
                         final_explanation.append(attention[i])
                final_explanation.append(0)
                attention = final_explanation
            if(rational==False):
                assert(len(attention) == len(row['Attention']))

            topk_indicies = sorted(range(len(attention)), key=lambda i: attention[i])[-topk:]

            temp_hard_rationales=[]
            for ind in topk_indicies:
                    temp_hard_rationales.append({'end_token':ind+1,'start_token':ind})

            temp["rationales"]=[{"docid": row['Post_id'], 
                                     "hard_rationale_predictions": temp_hard_rationales, 
                                     "soft_rationale_predictions": attention,
                                     #"soft_sentence_predictions":[1.0],
                                     "truth":0}]
            list_dict.append(temp)

       
    
    return list_dict,test_data


# In[115]:


def get_final_dict_with_lime(params,model_name,test_data,topk):
    list_dict_org,test_data=standaloneEval_with_lime(params,model_name,test_data=test_data, topk=topk)
    test_data_with_rational=convert_data(test_data,params,list_dict_org,rational_present=True,topk=topk)
    list_dict_with_rational,_=standaloneEval_with_lime(params,model_name,test_data=test_data_with_rational, topk=topk,rational=True)
    test_data_without_rational=convert_data(test_data,params,list_dict_org,rational_present=False,topk=topk)
    list_dict_without_rational,_=standaloneEval_with_lime(params,model_name,test_data=test_data_without_rational, topk=topk,rational=True)
    final_list_dict=[]
    for ele1,ele2,ele3 in zip(list_dict_org,list_dict_with_rational,list_dict_without_rational):
        ele1['sufficiency_classification_scores']=ele2['classification_scores']
        ele1['comprehensiveness_classification_scores']=ele3['classification_scores']
        final_list_dict.append(ele1)
    return final_list_dict
    return list_dict_org


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
    
    
    my_parser.add_argument('num_samples',
                           metavar='--number_of_samples',
                           type=int,
                           help='number of samples each instance of the data to pass in lime')
    
    
    my_parser.add_argument('attention_lambda',
                           metavar='--attention_lambda',
                           type=str,
                           help='required to assign the contribution of the atention loss')
    
    
    
    args = my_parser.parse_args()
    
    
    model_to_use=args.model_to_use
    
    params={}
    params['num_classes']=3
    params['data_file']=dict_data_folder[str(params['num_classes'])]['data_file']
    params['class_names']=dict_data_folder[str(params['num_classes'])]['class_label']
    temp_read=get_annotated_data(params)
    
    with open('Data/post_id_divisions.json', 'r') as fp:
        post_id_dict=json.load(fp)
    temp_read=temp_read[temp_read['post_id'].isin(post_id_dict['test'])]
    
    
    
    params=return_params(model_dict_params[model_to_use],float(args.attention_lambda))
    params['num_classes']=3
    params['num_samples']=args.num_samples
    params['variance']=1
    params['device']='cpu'
    fix_the_random(seed_val = params['random_seed'])
    test_data=get_test_data(temp_read,params,message='text')
    final_dict=get_final_dict_with_lime(params,model_to_use,test_data,topk=5)
    path_name=model_dict_params[model_to_use]
    path_name_explanation='explanations_dicts/'+path_name.split('/')[1].split('.')[0]+'_explanation_with_lime_'+str(params['num_samples'])+'_'+str(params['att_lambda'])+'.json'
    with open(path_name_explanation, 'w') as fp:
        fp.write('\n'.join(json.dumps(i,cls=NumpyEncoder) for i in final_dict))


# In[ ]:

