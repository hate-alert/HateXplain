import numpy as np

###this file contain different attention mask calculation from the n masks from n annotators. In this code there are 3 annotators

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def neg_softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(-(x - np.max(x)))
    return e_x / e_x.sum(axis=0)


###softmax attention
## input:
#### at mask: attention masks from 3 annotators,  
#### label: 
from numpy import array, exp


def sigmoid(z):
      g = 1 / (1 + exp(-z))
      return g



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

def aggregate_attention(at_mask,row,params):
    if(len(at_mask[0])==len(at_mask[1])==len(at_mask[2])):
        if(row['final_label'] in ['normal','non-toxic']):
            at_mask_fin=[1/len(at_mask[0]) for x in at_mask[0]]
        else:
            at_mask_fin=[]
       
            if(row['old_vs_new'] == 'old'):
                for i in range(0,3):
                    if(row['label'+str(i+1)] not in ['normal','non-toxic']):
                        at_mask_fin.append(at_mask[i])
            else:
                for i in range(0,2):
                    at_mask_fin.append(at_mask[i])

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
            elif(params['type_attention']=='raw'):
                pass
            
            elif(params['type_attention']=='individual'):
                pass
        if(params['decay']==True):
             at_mask_fin=decay(at_mask_fin,params)
            
        return at_mask_fin
    else:
        print("error at calculating attention")
        print(len(at_mask[0]),len(at_mask[1]),len(at_mask[2]))
        if(params['type_attention']=='individual'):
            return at_mask
        else:
            return at_mask[0]   
        
#         if(label=="normal"):
#             at_mask_fin=[1/len(at_mask[0]) for x in at_mask[0]]
#         else:
#             at_mask=params['variance']*at_mask
#             at_mask_fin=np.sum(at_mask,axis=0)
            
#             if(params['type_attention']=='sigmoid'):
#                 at_mask_fin=sigmoid(at_mask_fin)
#             elif (params['type_attention']=='softmax'):
#                 at_mask_fin=softmax(at_mask_fin)
#             elif (params['type_attention']=='neg_softmax'):
#                 at_mask_fin=neg_softmax(at_mask_fin)
#         if(params['decay']==True):
#              at_mask_fin=decay(at_mask_fin,params)
            
#         return at_mask_fin
#     else:
#         print("error at calculating attention")
#         print(len(at_mask[0]),len(at_mask[1]),len(at_mask[2]))
#         return []





















##softmax calculate 








    
    
# def softmax_attention(at_mask,label,variance=1):
#     if(len(at_mask[0])==len(at_mask[1])==len(at_mask[2])):
#         if(label=="normal"):
#             at_mask_fin=[1/len(at_mask[0]) for x in at_mask[0]]
#         else:
#             print(at_mask)
#             at_mask=variance*at_mask
#             print(at_mask)
#             at_mask_fin=np.sum(at_mask,axis=0)
#             print(at_mask_fin)
#             at_mask_fin=neg_softmax(at_mask_fin)
#         return at_mask_fin
#     else:
#         print("error at calculating attention")
#         print(len(at_mask[0]),len(at_mask[1]),len(at_mask[2]))
#         return []

    
# def sigmoid_attention(at_mask,label):
#     if(len(at_mask[0])==len(at_mask[1])==len(at_mask[2])):
#             at_mask_fin=np.sum(at_mask,axis=0)
#             at_mask_fin=sigmoid(at_mask_fin)
#             return at_mask_fin
#     else:
#         print("error at calculating attention")
#         print(len(at_mask[0]),len(at_mask[1]),len(at_mask[2]))
#         return []



if __name__ == '__main__':
    print(softmax_attention(np.array([[0,0,0],[0,1,0],[0,1,1]]),"offensive",1))