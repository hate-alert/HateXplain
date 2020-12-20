#!/bin/bash
num_samples=100
#python testing.py birnn $num_samples &
# python testing_with_lime.py bert_supervised $num_samples 0.001
# python testing_with_lime.py bert_supervised $num_samples 0.01
# python testing_with_lime.py bert_supervised $num_samples 0.1
# python testing_with_lime.py bert_supervised $num_samples 1.0
# python testing_with_lime.py bert_supervised $num_samples 10.0
# python testing_with_lime.py bert_supervised $num_samples 100.0


# python testing_with_lime.py bert_supervised $num_samples 100
# python testing_with_lime.py cnngru $num_samples 1
# python testing_with_lime.py birnn_scrat $num_samples 100
# python testing_with_rational.py birnn_scrat 100
# python testing_with_rational.py bert_supervised 100
python testing_with_lime.py bert $num_samples 1
python testing_with_lime.py birnn_att $num_samples 1
python testing_with_lime.py birnn $num_samples 1



# TEMPLATES
#python testing_with_lime.py birnn_att  $num_samples &
#python testing_with_lime.py birnn_scrat  $num_samples &
#python testing_with_lime.py cnngru  $num_samples &
#python testing_with_lime.py bert $num_samples &
#python testing_with_lime.py bert_supervised  $num_samples &
