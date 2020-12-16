#!/bin/bash
num_samples=100
#python testing.py birnn $num_samples &
python testing.py bert_supervised $num_samples 0.001
python testing.py bert_supervised $num_samples 0.01
python testing.py bert_supervised $num_samples 0.1
python testing.py bert_supervised $num_samples 1.0
python testing.py bert_supervised $num_samples 10.0
python testing.py bert_supervised $num_samples 100.0

#python testing.py birnn_att  $num_samples &
#python testing.py cnngru  $num_samples &
#python testing.py bert $num_samples &
#python testing.py bert_supervised  $num_samples &
