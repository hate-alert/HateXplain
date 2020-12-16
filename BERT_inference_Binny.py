import transformers 
import torch
# import neptune
# from knockknock import slack_sender

from api_config import project_name,proxies,api_token
import glob 
from transformers import get_linear_schedule_with_warmup
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
import random
from transformers import BertTokenizer
from bert_codes.feature_generation import combine_features,return_dataloader
from bert_codes.data_extractor import data_collector
from bert_codes.own_bert_models import *
from bert_codes.utils import *
from sklearn.metrics import accuracy_score,f1_score
from tqdm import tqdm
import os

from sklearn.metrics.pairwise import cosine_similarity
import numpy


if torch.cuda.is_available():    
    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


# neptune.init(project_name,api_token=api_token,proxies=proxies)
# neptune.set_project(project_name)

print("current gpu device", torch.cuda.current_device())
torch.cuda.set_device(0)
print("current gpu device",torch.cuda.current_device())




# webhook_url = "https://hooks.slack.com/services/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"
# @slack_sender(webhook_url=webhook_url, channel="#model_messages")
def Eval_phase(params,which_files='test',model=None):
    # if(which_files=='test'):
    #   test_files=glob.glob('full_data/Test/*.csv')
    # if(which_files=='train'):
    #   test_files=glob.glob('full_data/Train/*.csv')
    # if(which_files=='val'):
    #   test_files=glob.glob('full_data/Val/*.csv')
    
    '''Testing phase of the model'''
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained(params['path_files'], do_lower_case=False)

    if(params['is_model']==True):
        print("hello")
        model.eval()
    else:
        model=select_model(params['what_bert'],params['path_files'],params['weights'])
        model.cuda()
        model.eval()

    print(model)

    sent_1 = "I think  we should send them back"
    sent_2 = "europe is filled with muslimes"
    sent_3 = "love going to the movies and playing sports"

    input_ids = torch.tensor([tokenizer.encode(sent_1, add_special_tokens=True)]).cuda()  # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
    print(input_ids)
    with torch.no_grad():
        last_hidden_states = model.bert(input_ids)[1]  # Models outputs are now tuples
        print('*',last_hidden_states.shape)
    aa = last_hidden_states

    input_ids = torch.tensor([tokenizer.encode(sent_2, add_special_tokens=True)]).cuda()  # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
    print(input_ids)
    with torch.no_grad():
        last_hidden_states = model.bert(input_ids)[1]  # Models outputs are now tuples
        print('*',last_hidden_states.shape)
    bb = last_hidden_states

    input_ids = torch.tensor([tokenizer.encode(sent_3, add_special_tokens=True)]).cuda()  # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
    print(input_ids)
    with torch.no_grad():
        last_hidden_states = model.bert(input_ids)[1]  # Models outputs are now tuples
        print('*',last_hidden_states.shape)
    cc = last_hidden_states

    print(cosine_similarity(aa.cpu().reshape(1,-1), bb.cpu().reshape(1,-1)))
    print(cosine_similarity(aa.cpu().reshape(1,-1), cc.cpu().reshape(1,-1)))
    print(cosine_similarity(cc.cpu().reshape(1,-1), bb.cpu().reshape(1,-1)))
    # df_test=data_collector(test_files,params['language'],is_train=False,sample_ratio=params['sample_ratio'],type_train=params['how_train'])
    # sentences_test = df_test.text.values
    # labels_test = df_test.label.values
    # input_test_ids,att_masks_test=combine_features(sentences_test,tokenizer)
    # test_dataloader=return_dataloader(input_test_ids,labels_test,att_masks_test,batch_size=params['batch_size'],is_train=False)
    # print("Running Test...")
    # t0 = time.time()

    # # Put the model in evaluation mode--the dropout layers behave differently
    # # during evaluation.
    # # Tracking variables 
    # eval_loss, eval_accuracy = 0, 0
    # nb_eval_steps, nb_eval_examples = 0, 0
    # true_labels=[]
    # pred_labels=[]
    # # Evaluate data for one epoch
    # for batch in test_dataloader:
    #   # Add batch to GPU
    #   batch = tuple(t.to(device) for t in batch)
    #   # Unpack the inputs from our dataloader
    #   b_input_ids, b_input_mask, b_labels = batch
    #   # Telling the model not to compute or store gradients, saving memory and
    #   # speeding up validation
    #   with torch.no_grad():        
    #       outputs = model(b_input_ids, 
    #                       token_type_ids=None, 
    #                       attention_mask=b_input_mask)

    #   logits = outputs[0]
    #   # Move logits and labels to CPU
    #   logits = logits.detach().cpu().numpy()
    #   label_ids = b_labels.to('cpu').numpy()
    #   # Calculate the accuracy for this batch of test sentences.
    #   tmp_eval_accuracy = flat_accuracy(logits, label_ids)
    #   # Accumulate the total accuracy.
    #   eval_accuracy += tmp_eval_accuracy
        
    #   pred_labels+=list(np.argmax(logits, axis=1).flatten())
    #   true_labels+=list(label_ids.flatten())

    #   # Track the number of batches
    #   nb_eval_steps += 1

    # testf1=f1_score(true_labels, pred_labels, average='macro')
    # testacc=accuracy_score(true_labels,pred_labels)

    # if(params['logging']!='neptune' or params['is_model'] == True):
    #   # Report the final accuracy for this validation run.
    #   print(" Accuracy: {0:.2f}".format(testacc))
    #   print(" Fscore: {0:.2f}".format(testf1))
    #   print(" Test took: {:}".format(format_time(time.time() - t0)))
    # else:
    #   bert_model = params['path_files'][:-1]
    #   language  = params['language']
    #   name_one=bert_model+"_"+language
    #   neptune.create_experiment(name_one,params=params,send_hardware_metrics=False,run_monitoring_thread=False)
    #   neptune.append_tag(bert_model)
    #   neptune.append_tag(language)
    #   neptune.append_tag('test')
    #   neptune.log_metric('test_f1score',testf1)
    #   neptune.log_metric('test_accuracy',testacc)
    #   neptune.stop()
    
    # return testf1,testacc


params={
    'logging':'locals',
    'language':'English',
    'is_train':False,
    'is_model':False,
    'learning_rate':5e-5,
    'epsilon':1e-8,
    'path_files':'models_saved/multilingual_bert_English_all_multitask_own_100_only_bert',
    'sample_ratio':100,
    'how_train':'baseline',
    'epochs':5,
    'batch_size':16,
    'to_save':False,
    'weights':[1.0,1.0],
    'what_bert':'weighted',
    'save_only_bert':True
}




if __name__=='__main__':
    for lang in ['English']:#,'Polish','Portugese','German','Indonesian','Italian','Arabic']:
            params['language']=lang
            Eval_phase(params,'test')

