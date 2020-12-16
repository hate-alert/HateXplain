import torch
import torch.nn as nn

debug=False
# Custom Layers
class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)
        
        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0
        
        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)
        
        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))
        
    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        temp=x.contiguous().view(-1, feature_dim)
        if(debug):
            print("temp",temp.shape)
            print("weight",self.weight.shape)
        eij = torch.mm(temp, self.weight)
        if(debug):
            print("eij step 1",eij.shape)
        eij = eij.view(-1, step_dim)
        if(debug):
            print("eij step 2",eij.shape)
        if self.bias:
            eij = eij + self.b
        
        eij = torch.tanh(eij)
        
        eij[~mask] = float('-inf')
        a=torch.softmax(eij, dim=1)
        
#         a = torch.exp(eij)
#         if(debug==True):
#             print("a shape",a.shape)
#             print("mask shape",mask.shape)
#         if mask is not None:
#             a = a * mask

#         a = a /(torch.sum(a, 1, keepdim=True) + 1e-10)
        
        
        
        if(debug):
            print("attention",a.shape)
        
        weighted_input = x * torch.unsqueeze(a, -1)
        if(debug):
            print("weighted input",weighted_input.shape)
        
        return torch.sum(weighted_input, 1),a


class Attention_LBSA(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention_LBSA, self).__init__(**kwargs)
        
        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0
        
        weight = torch.zeros(feature_dim, feature_dim)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)
        context=torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(context)
        self.context_vector=nn.Parameter(context)
        if bias:
            self.b = nn.Parameter(torch.zeros(feature_dim))
        
    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        temp=x.contiguous().view(-1, feature_dim)
        if(debug):
            print("temp",temp.shape)
            print("weight",self.weight.shape)
        eij = torch.mm(temp, self.weight)
        if(debug):
            print("eij step 1",eij.shape)
        #eij = eij.view(-1, step_dim)
        if(debug):
            print("eij step 2",eij.shape)
        if self.bias:
            eij = eij + self.b
        eij = torch.tanh(eij)
        
        ### changedstep
        eij = torch.mm(eij, self.context_vector)
        if(debug):
            print("eij step 3",eij.shape)
            print("context_vector",self.context_vector.shape)
        eij = eij.view(-1, step_dim)

#         a = torch.exp(eij)
#         if(debug==True):
#             print("a shape",a.shape)
#             print("mask shape",mask.shape)
#         if mask is not None:
#             a = a * mask

#         a = a /(torch.sum(a, 1, keepdim=True) + 1e-10)

        eij[~mask] = float('-inf')
        a=torch.softmax(eij, dim=1)
        
        if(debug):
            print("attention",a.shape)
        
        weighted_input = x * torch.unsqueeze(a, -1)
        if(debug):
            print("weighted input",weighted_input.shape)
        
        return torch.sum(weighted_input, 1),a

class Attention_LBSA_sigmoid(Attention_LBSA):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super().__init__(feature_dim, step_dim, bias, **kwargs)
        
    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        temp=x.contiguous().view(-1, feature_dim)
        if(debug):
            print("temp",temp.shape)
            print("weight",self.weight.shape)
        eij = torch.mm(temp, self.weight)
        if(debug):
            print("eij step 1",eij.shape)
        #eij = eij.view(-1, step_dim)
        if(debug):
            print("eij step 2",eij.shape)
        if self.bias:
            eij = eij + self.b
        eij = torch.tanh(eij)
        
        ### changedstep
        eij = torch.mm(eij, self.context_vector)
        if(debug):
            print("eij step 3",eij.shape)
            print("context_vector",self.context_vector.shape)
        eij = eij.view(-1, step_dim)
        sigmoid = nn.Sigmoid()
        a=sigmoid(eij)
           
        if(debug==True):
            print("a shape",a.shape)
            print("mask shape",mask.shape)
        if mask is not None:
            a = a * mask

        if(debug):
            print("attention",a.shape)
        
        weighted_input = x * torch.unsqueeze(a, -1)
        if(debug):
            print("weighted input",weighted_input.shape)
        
        return torch.sum(weighted_input, 1),a

