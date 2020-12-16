### This is run when you want to select the parameters from the parameters file
from sklearn.model_selection import ParameterGrid
import json







params_data={
    'include_special':False,  #True is want to include <url> in place of urls if False will be removed 
    'bert_tokens':False, #True /False
    'type_attention':'softmax', #softmax
    'set_decay':0.1,
    'majority':2,
    'max_length':128,
    'variance':5,
    'window':4,
    'alpha':0.5,
    'p_value':0.8,
    'method':'additive',
    'decay':False,
    'normalized':False,
    'not_recollect':True,
}

#"birnn","birnnatt","birnnscrat","cnn_gru"


common_hp={
    'is_model':True,
    'logging':'neptune',  ###neptune /local
    'learning_rate':2e-5,  ### learning rate 2e-5 for bert 0.001 for gru
    'epsilon':1e-8,
    'batch_size':32,
    'to_save':True,
    'epochs':20,
    'auto_weights':True,
    'weights':[1.0,1.0,1.0],
    'model_name':'birnn',
    'random_seed':42,
    'max_length':128,
    'num_classes':3,
    'att_lambda':1,
    'device':'cuda',
    'train_att':True

}
    
    
params_bert={
    'path_files':'bert-base-uncased',
    'what_bert':'weighted',
    'save_only_bert':False,
    'supervised_layer_pos':11,
    'num_supervised_heads':1,
    'dropout_bert':0.1
 }


params_other = {
        "vocab_size": 0,
        "padding_idx": 0,
        "hidden_size":64,
        "embed_size":0,
        "embeddings":None,
        "drop_fc":0.2,
        "drop_embed":0.2,
        "drop_hidden":0.1,
        "train_embed":False,
        "seq_model":"gru",
        "attention":"softmax"
}


if(params_data['bert_tokens']):
    for key in params_other:
        params_other[key]='N/A'
else:
    for key in params_bert:
        params_bert[key]='N/A'


def Merge(dict1, dict2,dict3, dict4): 
    res = {**dict1, **dict2,**dict3, **dict4} 
    return res 

params = Merge(params_data,common_hp,params_bert,params_other)


if __name__=='__main__':  
    params_list = []
    params_new = {}
    for key in params.keys():
        params_new[key]=[params[key]]
    
   
    params_new['model_name']=["birnnscrat"]
    params_new['learning_rate']=[0.1,0.01,0.001]
    params_new['hidden_size']=[64,128]
    params_new['drop_embed'] = [0.1,0.2,0.5]
    params_new['drop_fc'] = [0.1,0.2,0.5]
    params_new['att_lambda']=[0.001,0.01,0.1,1,10,100]
    #params_new['drop_hidden'] = [0.1,0.2,0.5]
    params_new['seq_model']=['lstm','gru']
    params_new['train_embed']=[True,False]
    params_list=list(ParameterGrid(params_new))
    print('Total experiments to be done:',len(params_list))
    
    with open('all_params_scrat.json', 'w') as fout:
            json.dump(params_list ,fout,indent=4)
        










# for train_att in [True,False]:
#         print(train_att)
#         params['train_att']=train_att
#         if(train_att):
#             for supervised_layer_pos in range(10,12):
#                 params['supervised_layer_pos'] = supervised_layer_pos
#                 for num_supervised_heads in range(10,12):
#                     params['num_supervised_heads']= num_supervised_heads 
#                     for att_lambda in [0.01,0.1,1,10,100]:
#                         params['att_lambda']=att_lambda
#                         for dropout_bert in [0.1,0.5]:
#                             params['dropout_bert']=dropout_bert
#                             for auto_weights in [True,False]:
#                                 params['auto_weights']=auto_weights
#                                 for learning_rate in [2e-5]:
#                                     params['learning_rate']=learning_rate
#                                     params_list.append(params.copy())
#         else:
#             for dropout_bert in [0.1,0.5]:
#                 params['dropout_bert']=dropout_bert
#                 for auto_weights in [True,False]:
#                     params['auto_weights']=auto_weights
#                     for learning_rate in [2e-5]:
#                         params['learning_rate']=learning_rate
#                         params_list.append(params.copy())
# #     


























