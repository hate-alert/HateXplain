### Parameters used in this code
This document notes down all the hyper-parameters associated with data, models and evaluation. Before adding any new options to some of the categories, please ensure that is implemented in the respective python file.
Parameters other than the ones described below are to be kept fixed.
##### Attention aggregation parameters
* **type_attention** :- How the normalisation of the attention vector will happen. Three options are available currently "softmax","neg_softmax" and "sigmoid". More details [here](https://github.com/punyajoy/HateXplain/blob/master/Preprocess/attentionCal.py) 
"variance": 5.0,

##### Preprocessing parameters

"include_special": "False",
"max_length": 128.0,
"padding_idx": 0.0,

#### Miscellanous
"bert_tokens": "False",
"logging": "neptune",
"model_name": "birnnscrat"
"num_classes": 3.0,
"random_seed": 42.0,
"to_save": "True",

##### Non-BERT model parameters
"drop_embed": 0.5,
"drop_fc": 0.2,
"drop_hidden": 0.1,
"embed_size": 300.0,
"hidden_size": 64.0,
"seq_model": "lstm",

##### BERT model parameters
"dropout_bert": "N/A",
"num_supervised_heads": "N/A",
"what_bert": "N/A",
"save_only_bert": "N/A",
"supervised_layer_pos": "N/A",

##### Common parameters for training 
"att_lambda": 100.0,
"auto_weights": "True",
"batch_size": 32.0,
"device": "cuda",
"embeddings": "None",
"epochs": 20.0,
"epsilon": 1e-08,
"learning_rate": 0.001,
"weights": "[1.0795518  0.82139814 1.1678787 ]",
"train_att": "True",
"train_embed": "True",

"normalized": "False",




