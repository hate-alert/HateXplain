#!/bin/bash
# python manual_training_inference.py best_model_json/bestModel_bert_base_uncased_Attn_train_FALSE.json True &
# sleep 15s
python manual_training_inference.py best_model_json/bestModel_bert_base_uncased_Attn_train_TRUE.json True 0.001 &
sleep 15s
python manual_training_inference.py best_model_json/bestModel_bert_base_uncased_Attn_train_TRUE.json True 0.01 &
sleep 15s
python manual_training_inference.py best_model_json/bestModel_bert_base_uncased_Attn_train_TRUE.json True 0.1 &
sleep 15s
python manual_training_inference.py best_model_json/bestModel_bert_base_uncased_Attn_train_TRUE.json True 1 &
sleep 15s
python manual_training_inference.py best_model_json/bestModel_bert_base_uncased_Attn_train_TRUE.json True 10 &
sleep 15s
python manual_training_inference.py best_model_json/bestModel_bert_base_uncased_Attn_train_TRUE.json True 100 &
sleep 15s

# python manual_training_inference.py best_model_json/bestModel_birnn.json True &
#python manual_training_inference.py best_model_json/bestModel_birnnscrat.json True &
# python manual_training_inference.py best_model_json/bestModel_birnnscrat.json True 0.01 &
# python manual_training_inference.py best_model_json/bestModel_birnnscrat.json True 0.1 &
# python manual_training_inference.py best_model_json/bestModel_birnnscrat.json True 1 &
# python manual_training_inference.py best_model_json/bestModel_birnnscrat.json True 10 &
# python manual_training_inference.py best_model_json/bestModel_birnnscrat.json True 100 &

# sleep 15s
# python manual_training_inference.py best_model_json/bestModel_birnnatt.json True &
# sleep 15s
# python manual_training_inference.py best_model_json/bestModel_cnn_gru.json True &
