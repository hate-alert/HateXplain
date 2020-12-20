import numpy as np
from numpy import array, exp
###this file contain different attention mask calculation from the n masks from n annotators. In this code there are 3 annotators



#### Few helper functions to convert attention vectors in 0 to 1 scale. While softmax converts all the values such that their sum lies between 0 --> 1. Sigmoid converts each value in the vector in the range 0 -> 1.

##### We mostly use softmax
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
 
def neg_softmax(x):
    """Compute softmax values for each sets of scores in x. Here we convert the exponentials to 1/exponentials"""
    e_x = np.exp(-(x - np.max(x)))
    return e_x / e_x.sum(axis=0)
def sigmoid(z):
      """Compute sigmoid values"""
      g = 1 / (1 + exp(-z))
      return g


##### This function is used to aggregate the attentions vectors. This has a lot of options refer to the parameters explanation for understanding each parameter.
def aggregate_attention(at_mask,row,params):
    """input: attention vectors from 2/3 annotators (at_mask), row(dataframe row), params(parameters_dict)
       function: aggregate attention from different annotators.
       output: aggregated attention vector"""
    
    
    #### If the final label is normal or non-toxic then each value is represented by 1/len(sentences)
    if(row['final_label'] in ['normal','non-toxic']):
        at_mask_fin=[1/len(at_mask[0]) for x in at_mask[0]]
    else:
        at_mask_fin=at_mask
        #### Else it will choose one of the options, where variance is added, mean is calculated, finally the vector is normalised.   
        if(params['type_attention']=='sigmoid'):
            at_mask_fin=int(params['variance'])*at_mask_fin
            at_mask_fin=np.mean(at_mask_fin,axis=0)
            at_mask_fin=sigmoid(at_mask_fin)
        elif (params['type_attention']=='softmax'):
            at_mask_fin=int(params['variance'])*at_mask_fin
            at_mask_fin=np.mean(at_mask_fin,axis=0)
            at_mask_fin=softmax(at_mask_fin)
        elif (params['type_attention']=='neg_softmax'):
            at_mask_fin=int(params['variance'])*at_mask_fin
            at_mask_fin=np.mean(at_mask_fin,axis=0)
            at_mask_fin=neg_softmax(at_mask_fin)
        elif(params['type_attention'] in ['raw','individual']):
            pass
    if(params['decay']==True):
         at_mask_fin=decay(at_mask_fin,params)

    return at_mask_fin
    
    
    
##### Decay and distribution functions.To decay the attentions left and right of the attented word. This is done to decentralise the attention to a single word. 
def distribute(old_distribution, new_distribution, index, left, right,params):
    window = params['window']
    alpha = params['alpha']
    p_value = params['p_value']
    method =params['method']
    
    
    reserve = alpha * old_distribution[index]
#     old_distribution[index] = old_distribution[index] - reserve
    
    if method=='additive':    
        for temp in range(index - left, index):
            new_distribution[temp] = new_distribution[temp] + reserve/(left+right)
        
        for temp in range(index + 1, index+right):
            new_distribution[temp] = new_distribution[temp] + reserve/(left+right)
    
    if method == 'geometric':
        # we first generate the geometric distributio for the left side
        temp_sum = 0.0
        newprob = []
        for temp in range(left):
            each_prob = p_value*((1.0-p_value)**temp)
            newprob.append(each_prob)
            temp_sum +=each_prob
            newprob = [each/temp_sum for each in newprob]
        
        for temp in range(index - left, index):
            new_distribution[temp] = new_distribution[temp] + reserve*newprob[-(temp-(index-left))-1]
        
        # do the same thing for right, but now the order is opposite
        temp_sum = 0.0
        newprob = []
        for temp in range(right):
            each_prob = p_value*((1.0-p_value)**temp)
            newprob.append(each_prob)
            temp_sum +=each_prob
            newprob = [each/temp_sum for each in newprob]
        for temp in range(index + 1, index+right):
            new_distribution[temp] = new_distribution[temp] + reserve*newprob[temp-(index + 1)]
    
    return new_distribution



def decay(old_distribution,params):
    window=params['window']
    new_distribution = [0.0]*len(old_distribution)
    for index in range(len(old_distribution)):
        right = min(window, len(old_distribution) - index)
        left = min(window, index)
        new_distribution = distribute(old_distribution, new_distribution, index, left, right, params)

    if(params['normalized']):
        norm_distribution = []
        for index in range(len(old_distribution)):
            norm_distribution.append(old_distribution[index] + new_distribution[index])
        tempsum = sum(norm_distribution)
        new_distrbution = [each/tempsum for each in norm_distribution]
    return new_distribution


