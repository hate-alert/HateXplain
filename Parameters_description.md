### Parameters used in this code
This document notes down all the hyper-parameters associated with data, models and evaluation. Before adding any new options to some of the categories, please ensure that is implemented in the respective python file.
Parameters other than the ones described below are to be kept fixed.
##### Attention aggregation parameters
* **type_attention** :- How the normalisation of the attention vector will happen. Three options are available currently "softmax","neg_softmax" and "sigmoid". More details [here](https://github.com/punyajoy/HateXplain/blob/master/Preprocess/attentionCal.py).
* **variance**:- constant multiplied with the attention vector to increase the difference between the attention to attended and non-attended tokens.More details [here](https://github.com/punyajoy/HateXplain/blob/master/Preprocess/attentionCal.py). 

~~~
variance=5
attention = [0,0,0,1,0]
attention_modified = attention * variance
attention_modified = [0,0,0,5,0]
~~~

##### Preprocessing parameters
* **include_special** :- This can be set as *True* or *False*. This is with respect to [ekaphrasis](https://github.com/cbaziotis/ekphrasis) processing. For example ekphrasis adds a special character to the hashtags after processing. for e.g. #happyholidays will be transformed to <hashtag> happy holidays </hashtag>. If set the `include specials` to false than the begin token and ending token will be removed. 
* **max length** :- This represent the maximum length of the words in case of non transformers models or subwords in case of transformers models. For all our models this is set to 128.

#### Miscellanous
* **bert_tokens**: This can be set as *True* or *False*. This is the main condition that decides whether the final model and preprocessing will be transformer based (`bert_tokens` to True) or non-transformer based (`bert_tokens` to False).
* **logging**: This can be set as *local* or *neptune*. Setting the  `logging` as "neptune" is used to plot the metrics to [neptune](https://neptune.ai/) and experiment logger platform. See setup instruction in the webpage for more information. keep `logging` as "local" to not use it.
* **model_name**: These are used for the non transformer models mainly. In the 
* **num_classes**: These represent the number of classes. This is `3` for our case except for bias evaluation where it is 2. 
* **random_seed**: This should be set to `42` for reproducibility.
* **to_save**: This can be set as *True* or *False*. It controls if you want to save the final model or not.

##### Non-BERT model parameters
* **drop_embed**: Dropout after the embedding layer,
* **drop_fc**: Dropout after the fully connected layer,
* **embed_size**: Embedding size that needs to be used. For our purpose we set it to 300. More info in the Preprocess section
* **hidden_size**: Hidden size for the recurrent neural network,
* **seq_model**: Which recurrence model to use "lstm" or "gru",
* **train_embed**: This can be set as *True* or *False*. This will be set as True if you want to train the embeddings layer also.

##### BERT model parameters
* **dropout_bert**: Dropout after the linear layer in BERT,
* **num_supervised_heads**: Number of attention heads whose final attention needs to be aligned with the ground truth attention,
* **save_only_bert**: This can be set as *True* or *False*. "True" will save the BERT part of the model only not the linear layer.
* **supervised_layer_pos**: The layer whose attention heads needs to be aligned in the final layer,

##### Common parameters for training 
* **att_lambda**: Contribution of the attention loss to the total loss.
* **auto_weights**: This can be set as *True* or *False*. True will assign the class weight based on the class distribution in the training dataset.
* **batch_size**: Batch size to train the model. We set it to 32 for every model.
* **device**: Device on which the system will run. set "cuda(gpu)" or "cpu".
* **epochs**: Number of epochs to train the model.
* **epsilon**: Used as a parameter in Adam optimizer. Default set at 1e-08,
* **learning_rate**: Learning rate passed to the Adam optimizer. For BERT it is set to closer to 2e-5, for non-transformer model it is in the range of 0.001 to 0.1 in our case. 
* **weights**: If you want to manually set the weights for the different classes. Be sure to maintain a vector of length similar to the original number of classes. 
* **train_att**: This can be set as *True* or *False*. This will be set as True if you want to train the attention weights of the model.

