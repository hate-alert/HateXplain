# import transformers 
# import torch
# import neptune
# from knockknock import slack_sender

# from api_config import project_name,proxies,api_token,project_name_test
# import glob 
# from transformers import BertTokenizer
# from transformers import BertForSequenceClassification, AdamW, BertConfig
# import random
# from transformers import BertTokenizer
# from bert_codes.feature_generation import combine_features,return_dataloader
# from bert_codes.data_extractor import data_collector
# from bert_codes.own_bert_models import *
# from bert_codes.utils import *
# from sklearn.metrics import accuracy_score,f1_score
# from tqdm import tqdm
# import os

# from pandas_ml import ConfusionMatrix
# import matplotlib.pyplot as plt





# def Eval_phase(params,which_files='test',model=None):
# 	# if(params['language']=='English'):
# 	# 	params['csv_file']='*_full.csv'
	

# 	if(which_files=='train'):
# 		path=params['files']+'/Train'+params['csv_file']
# 		test_files=glob.glob(path)
# 	if(which_files=='val'):
# 		path=params['files']+'/Val'+params['csv_file']
# 		test_files=glob.glob(path)
# 	if(which_files=='test'):
# 		path=params['files']+'/Test'+params['csv_file']
# 		test_files=glob.glob(path)
	
# 	'''Testing phase of the model'''
# 	print('Loading BERT tokenizer...')
# 	tokenizer = BertTokenizer.from_pretrained(params['path_files'], do_lower_case=False)

# 	if(params['is_model']==True):
# 		print("model previously passed")
# 		model.eval()
# 	else:
# 		model=select_model(params['what_bert'],params['path_files'],params['weights'])
# 		model.cuda()
# 		model.eval()

		
# 	#df_test=data_collector(test_files,params,False)
# 	df_test=pd.read_csv(test_files[0])
# 	if(params['csv_file']=='*_translated.csv'):
# 		sentences_test = df_test.translated.values
# 	elif(params['csv_file']=='.csv'):
# 		sentences_test = df_test.Text.values
		

# 	labels_test = df_test.label.values
# 	input_test_ids,att_masks_test=combine_features(sentences_test,tokenizer,params['max_length'])
# 	test_dataloader=return_dataloader(input_test_ids,labels_test,att_masks_test,batch_size=params['batch_size'],is_train=False)
# 	print("Running eval on ",which_files,"...")
# 	t0 = time.time()

# 	# Put the model in evaluation mode--the dropout layers behave differently
# 	# during evaluation.
# 	# Tracking variables 
# 	eval_loss, eval_accuracy = 0, 0
# 	nb_eval_steps, nb_eval_examples = 0, 0
# 	true_labels=RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.FloatTensor [16, 128]], which is output 0 of SoftmaxBackward, is at version 1; expected version 0 instead. Hint: the backtrace further above shows the operation that failed to compute its gradient. The variable in question was changed in there or anywhere later. Good luck!
# []
# 	pred_labels=[]
# 	# Evaluate data for one epoch
# 	for batch in test_dataloader:
# 		# Add batch to GPU
# 		batch = tuple(t.to(device) for t in batch)
# 		# Unpack the inputs from our dataloader
# 		b_input_ids, b_input_mask, b_labels = batch
# 		# Telling the model not to compute or store gradients, saving memory and
# 		# speeding up validation
# 		with torch.no_grad():        
# 			outputs = model(b_input_ids, 
# 							token_type_ids=None, 
# 							attention_mask=b_input_mask)

# 		logits = outputs[0]
# 		# Move logits and labels to CPU
# 		logits = logits.detach().cpu().numpy()
# 		label_ids = b_labels.to('cpu').numpy()
# 		# Calculate the accuracy for this batch of test sentences.
# 		tmp_eval_accuracy = flat_accuracy(logits, label_ids)
# 		# Accumulate the total accuracy.
# 		eval_accuracy += tmp_eval_accuracy
		
# 		pred_labels+=list(np.argmax(logits, axis=1).flatten())
# 		true_labels+=list(label_ids.flatten())

# 		# Track the number of batches
# 		nb_eval_steps += 1

# 	testf1=f1_score(true_labels, pred_labels, average='macro')
# 	testacc=accuracy_score(true_labels,pred_labels)

# 	if(params['logging']!='neptune' or params['is_model'] == True):
# 		# Report the final accuracy for this validation run.
# 		print(" Accuracy: {0:.2f}".format(testacc))
# 		print(" Fscore: {0:.2f}".format(testf1))
# 		print(" Test took: {:}".format(format_time(time.time() - t0)))
# 		print(ConfusionMatrix(true_labels,pred_labels))
# 	else:
# 		bert_model = params['path_files']
# 		language  = params['language']
# 		name_one=bert_model+"_"+language
# 		neptune.create_experiment(name_one,params=params,send_hardware_metrics=False,run_monitoring_thread=False)
# 		neptune.append_tag(bert_model)
# 		neptune.append_tag(language)
# 		neptune.append_tag('test')
# 		neptune.log_metric('test_f1score',testf1)
# 		neptune.log_metric('test_accuracy',testacc)
# 		neptune.stop()
	
# 	return testf1,testacc


# params={
# 	'logging':'neptune',
# 	'language':'English',
# 	'is_train':True,
# 	'is_model':False,
# 	'learning_rate':2e-5,
# 	'files':'only_hate',
# 	'csv_file':'*_translated.csv',
# 	'samp_strategy':'stratified',
# 	'epsilon':1e-8,
# 	'path_files':'multilingual_bert',
# 	'take_ratio':False,
# 	'sample_ratio':16,
# 	'how_train':'baseline',
# 	'epochs':5,
# 	'batch_size':16,
# 	'to_save':True,
# 	'weights':[1.0,1.0],
# 	'what_bert':'normal',
# 	'save_only_bert':False,
# 	'max_length':128,
# 	'columns_to_consider':['directness','target','group'],
# 	'random_seed':42

# }

# if __name__=='__main__':
    

#     if torch.cuda.is_available():    
#         # Tell PyTorch to use the GPU.    
#         device = torch.device("cuda")
#         print('There are %d GPU(s) available.' % torch.cuda.device_count())
#         print('We will use the GPU:', torch.cuda.get_device_name(0))
#     # If not...
#     else:
#         print('No GPU available, using the CPU instead.')
#         device = torch.device("cpu")


# 	neptune.init(project_name_test,api_token=api_token,proxies=proxies)
# 	neptune.set_project(project_name_test)

# 	lang_map={'Arabic':'ar','French':'fr','Portugese':'pt','Spanish':'es','English':'en','Indonesian':'id','Italian':'it','German':'de','Polish':'pl'}
# 	torch.cuda.set_device(0)

# 	lang_list=list(lang_map.keys())
# 	for lang in lang_list:
# 		if(lang=='English'):
# 			continue
# 		params['language']=lang
# 		for sample_ratio,take_ratio in [(16,False),(32,False),(64,False),(128,False),(256,False)]:
# 			count=0
# 			params['take_ratio']=take_ratio
# 			params['sample_ratio']=sample_ratio
# 			if(params['csv_file']=='*_full.csv'):
# 				translate='translated'
# 			else:
# 				translate='actual'
# 			# Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
# 			if(params['how_train']!='all'):
# 				output_dir = 'models_saved/'+params['path_files']+'_'+params['language']+'_'+translate+'_'+params['how_train']+'_'+str(params['sample_ratio'])
# 			else:
# 				output_dir = 'models_saved/'+params['path_files']+'_'+translate+'_'+params['how_train']+'_'+str(params['sample_ratio'])
			

# 			temp_path=params['path_files']
# 			params['path_files']=output_dir 
# 			print(output_dir)
# 			Eval_phase(params,'test')
# 			params['path_files']=temp_path




	
