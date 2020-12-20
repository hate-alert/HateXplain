import pandas as pd
from glob import glob
import json
from tqdm import tqdm_notebook,tqdm
from difflib import SequenceMatcher
from collections import Counter
from .preProcess import ek_extra_preprocess
from .attentionCal import aggregate_attention
from .spanMatcher import returnMask,returnMaskonetime
from transformers import BertTokenizer
from .utils import CheckForGreater,most_frequent
from .preProcess import *   
from transformers import BertTokenizer
from os import path
import pickle
import numpy as np
import re

def set_name(params):
    file_name='Data/Total_data'
    if(params['bert_tokens']):
        file_name+='_bert'
    else:
        file_name+='_normal'
    
    file_name+='_'+params['type_attention']+'_'+str(params['variance'])+'_'+str(params['max_length'])
    if(params['decay']):
        file_name+='_'+params['method']+'_'+str(params['window'])+'_'+str(params['alpha']) +'_'+str(params['p_value'])    
    file_name+='_'+str(params['num_classes'])+'.pickle'
    return file_name



def get_annotated_data(params):
    #temp_read = pd.read_pickle(params['data_file'])
    with open(params['data_file'], 'r') as fp:
        data = json.load(fp)
    dict_data=[]
    for key in data:
        temp={}
        temp['post_id']=key
        temp['text']=data[key]['post_tokens']
        final_label=[]
        for i in range(1,4):
            temp['annotatorid'+str(i)]=data[key]['annotators'][i-1]['annotator_id']
#             temp['explain'+str(i)]=data[key]['annotators'][i-1]['rationales']
            temp['target'+str(i)]=data[key]['annotators'][i-1]['target']
            temp['label'+str(i)]=data[key]['annotators'][i-1]['label']
            final_label.append(temp['label'+str(i)])

        final_label_id=max(final_label,key=final_label.count)
        temp['rationales']=data[key]['rationales']
            
        if(params['class_names']=='Data/classes_two.npy'):
            if(final_label.count(final_label_id)==1):
                temp['final_label']='undecided'
            else:
                if(final_label_id in ['hatespeech','offensive']):
                    final_label_id='toxic'
                else:
                    final_label_id='non-toxic'
                temp['final_label']=final_label_id

        
        else:
            if(final_label.count(final_label_id)==1):
                temp['final_label']='undecided'
            else:
                temp['final_label']=final_label_id

        
        
        
        dict_data.append(temp)    
    temp_read = pd.DataFrame(dict_data)  
    return temp_read    





def get_training_data(data,params,tokenizer):
    '''input: data is a dataframe text ids attentions labels column only'''
    '''output: training data in the columns post_id,text, attention and labels '''

    majority=params['majority']
    post_ids_list=[]
    text_list=[]
    attention_list=[]
    label_list=[]
    count=0
    count_confused=0
    print('total_data',len(data))
    for index,row in tqdm(data.iterrows(),total=len(data)):
        #print(params)
        text=row['text']
        post_id=row['post_id']

        annotation_list=[row['label1'],row['label2'],row['label3']] 
        annotation=row['final_label']
        
        if(annotation != 'undecided'):
            tokens_all,attention_masks=returnMask(row,params,tokenizer)
            attention_vector= aggregate_attention(attention_masks,row, params)     
            attention_list.append(attention_vector)
            text_list.append(tokens_all)
            label_list.append(annotation)
            post_ids_list.append(post_id)
        else:
            count_confused+=1
            
    print("attention_error:",count)
    print("no_majority:",count_confused)
    # Calling DataFrame constructor after zipping 
    # both lists, with columns specified 
    training_data = pd.DataFrame(list(zip(post_ids_list,text_list,attention_list,label_list)), 
                   columns =['Post_id','Text', 'Attention' , 'Label']) 
    
    
    filename=set_name(params)
    training_data.to_pickle(filename)
    return training_data

##### Data collection for test data
def get_test_data(data,params,message='text'):
    '''input: data is a dataframe text ids labels column only'''
    '''output: training data in the columns post_id,text (tokens) , attentions (normal) and labels'''
    
    if(params['bert_tokens']):
        print('Loading BERT tokenizer...')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=False)
    else:
        tokenizer=None
    
    
    post_ids_list=[]
    text_list=[]
    attention_list=[]
    label_list=[]
    print('total_data',len(data))
    for index,row in tqdm(data.iterrows(),total=len(data)):
        post_id=row['post_id']
        annotation=row['final_label']
        tokens_all,attention_masks=returnMask(row,params,tokenizer)
        attention_vector= aggregate_attention(attention_masks,row, params) 
        attention_list.append(attention_vector)
        text_list.append(tokens_all)
        label_list.append(annotation)
        post_ids_list.append(post_id)
    
    
    # Calling DataFrame constructor after zipping 
    # both lists, with columns specified 
    training_data = pd.DataFrame(list(zip(post_ids_list,text_list,attention_list,label_list)), 
                   columns =['Post_id','Text', 'Attention' , 'Label']) 
    
    
    return training_data







def convert_data(test_data,params,list_dict,rational_present=True,topk=2):
    """this converts the data to be with or without the rationals based on the previous predictions"""
    """input: params -- input dict, list_dict -- previous predictions containing rationals
    rational_present -- whether to keep rational only or remove them only
    topk -- how many words to select"""
    
    temp_dict={}
    for ele in list_dict:
        temp_dict[ele['annotation_id']]=ele['rationales'][0]['soft_rationale_predictions']
    
    test_data_modified=[]
    
    for index,row in tqdm(test_data.iterrows(),total=len(test_data)):
        try:
            attention=temp_dict[row['Post_id']]
        except KeyError:
            continue
        topk_indices = sorted(range(len(attention)), key=lambda i: attention[i])[-topk:]
        new_text =[]
        new_attention =[]
        if(rational_present):
            if(params['bert_tokens']):
                new_attention =[0]
                new_text = [101]
            for i in range(len(row['Text'])):
                if(i in topk_indices):
                    new_text.append(row['Text'][i])
                    new_attention.append(row['Attention'][i])
            if(params['bert_tokens']):
                new_attention.append(0)
                new_text.append(102)
        else:
            for i in range(len(row['Text'])):
                if(i not in topk_indices):
                    new_text.append(row['Text'][i])
                    new_attention.append(row['Attention'][i])
        test_data_modified.append([row['Post_id'],new_text,new_attention,row['Label']])

    df=pd.DataFrame(test_data_modified,columns=test_data.columns)
    return df



def transform_dummy_data(sentences):
    post_id_list=['temp']*len(sentences)
    pred_list=['normal']*len(sentences)
    explanation_list=[]
    sentences_list=[]
    for i in range(len(sentences)):
        explanation_list.append([])
        sentences_list.append(sentences[i].split(" "))
    df=pd.DataFrame(list(zip(post_id_list,sentences_list,pred_list,pred_list,
                             pred_list,explanation_list,pred_list)),
                         columns=['post_id', 'text', 'label1','label2','label3', 'rationales', 'final_label'])
    
    return df


def collect_data(params):
    if(params['bert_tokens']):
        print('Loading BERT tokenizer...')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=False)
    else:
        tokenizer=None
    data_all_labelled=get_annotated_data(params)
    train_data=get_training_data(data_all_labelled,params,tokenizer)
    return train_data
























##### ONLY FOR ONE TIME USAGE
def get_training_data_one_time(data,params,tokenizer):
    '''input: data is a dataframe text ids attentions labels column only'''
    '''output: training data in the columns post_id,text, attention and labels '''

    majority=params['majority']
    
    dict_data={}
   
    
    print('total_data',len(data))
    
    for index,row in tqdm(data.iterrows(),total=len(data)):
        post_id=row['post_id']
        dict_data[post_id]={}
        dict_data[post_id]['post_id']=post_id
        dict_data[post_id]['annotators']=[]
        #dict_data[post_id]['old_vs_new']=row['old_vs_new']
        selection_list=[row['explain1'],row['explain2'],row['explain3']]
        tokens_all,attention_masks,string_parts,list_pos,span_list,list_mask \
        =returnMaskonetime(row,params,tokenizer,data_type=row['old_vs_new'])
        temp_users=[]
        
        rational_list=[]
        
        for i in range(1,4):
            temp_user_dict={}
            temp_user_dict['label']=row['pred'+str(i)]
            #temp_user_dict['rationales']=attention_masks[i-1]
            if(row['final_annotation'] not in ['normal','non-toxic','undecided']):
                if(row['old_vs_new']=='new' and i<3):
                    rational_list.append(attention_masks[i-1])
                elif(row['old_vs_new']=='old' and (row['pred'+str(i)] not in ['normal','non-toxic'])):
                    rational_list.append(attention_masks[i-1])

            temp_user_dict['annotator_id']=row['workerid'+str(i)]
            temp_user_dict['target']=row['target'+str(i)]
            temp_users.append(temp_user_dict)
            
            
        dict_data[post_id]['rationales']=rational_list
        dict_data[post_id]['post_tokens']=tokens_all
        dict_data[post_id]['annotators']=temp_users
    
    return dict_data        





#### OLDcode remove at last
def return_inverse_dict():
    with open("../../main/id_orig_seid_Mapping.json") as f:
        id_dict_orig = json.load(f) 



    orig_dict_id={}
    for key in tqdm(id_dict_orig.keys()):    
        orig_text=id_dict_orig[key][0]
        seid_text=id_dict_orig[key][1]

        orig_dict_id[seid_text]=[key,orig_text]
    return orig_dict_id


def return_id_orig(text,orig_dict_id):
    try:
        #to return the test directly 
        return orig_dict_id[text][0],orig_dict_id[text][1]
    except:
        max_sim=0
        max_text=""
        for key in orig_dict_id.keys():
            text_id=orig_dict_id[key][0]
            orig_text=orig_dict_id[key][1]
            sim=similar(key,text)
            if(sim>max_sim):
                max_sim=sim
                max_text=key
            if(sim>0.95):
                return text_id,orig_text
        print(text,"||",max_text,"||",max_sim)
        return -1,-1

    
def get_text_information(df,key,text_id_map):
    dict_text={}
    text='Input.text'
    gender='Answer.Gender'
    miscellanous='Answer.Miscellaneous'
    origin='Answer.Origin'
    race='Answer.Race'
    sexual='Answer.Sexual'
    annotated='Answer.sentiment'
    religion='Answer.Religion'
    selection ='Answer.selections'
    for i in range(1,7):
        if(i>1):
            own_text=text+str(i)
            own_gender=gender+str(i)
            own_miscl=miscellanous+str(i)
            own_origin=origin+str(i)
            own_race=race+str(i)
            own_sexual=sexual+str(i)
            own_annotated=annotated+str(i)
            own_religion=religion+str(i)
            
        else:
            own_text=text
            own_gender=gender
            own_miscl=miscellanous
            own_origin=origin
            own_race=race
            own_sexual=sexual
            own_annotated=annotated
            own_religion=religion
        own_selection = selection+str(i)
        if(df.iloc[0][own_text]==df.iloc[1][own_text]==df.iloc[2][own_text]):
            
            
            id_text,orig_text=return_id_orig(df.iloc[0][own_text],text_id_map)
            if(id_text==-1):
                id_text=key+"_nf_"+str(i)
                orig_text=cleanhtml(df.iloc[0][own_text])
                print(id_text)
            
            dict_text[id_text]={'text':orig_text}
            for k in range(0,3):
                str_user='user'+str(k+1)
                dict_text[str_user]={
                                        'annotation':df.iloc[k][own_annotated],
                                        'gender':df.iloc[k][own_gender],
                                        'miscellanous':df.iloc[k][own_miscl],
                                        'origin':df.iloc[k][own_origin],
                                        'race':df.iloc[k][own_race],
                                        'sexual':df.iloc[k][own_sexual],  
                                        'religion':df.iloc[k][own_religion],
                                        'selection':df.iloc[k][own_selection]
                                    }
            
            
    
    
    return dict_text
    
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()
    
def get_dict_comments(orig_dict_id,data_all_labelled,params):
    dict_comment_file = 'dict_comments.dat'
    not_rerun=params['not_recollect']
    if(path.isfile(dict_comment_file) and not_rerun):
        with open(dict_comment_file, 'rb') as handle:
            dict_comments = pickle.load(handle)
    else:
        print('hello')
        dict_comments={}
        data_all_labelled=data_all_labelled.reset_index(drop=True) 
        grouped_data=data_all_labelled.groupby("HITId")
        grouped_data=dict(list(grouped_data))
        for key in tqdm(grouped_data.keys()):
            dict_new=get_text_information(grouped_data[key],key,orig_dict_id)
            for key in dict_new.keys():
                dict_comments[key]=dict_new[key]

        with open(dict_comment_file, 'wb') as handle:
            pickle.dump(dict_comments, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return dict_comments


