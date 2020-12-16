import torch
import torch.nn as nn
import numpy as np
from Models.attentionLayer import *
from .utils import masked_cross_entropy
debug =False
#### BiGRUCLassifier model

def global_max_pooling(tensor, dim, topk):
    """Global max pooling"""
    ret, _ = torch.topk(tensor, topk, dim)
    return ret



class BiRNN(nn.Module):  
    def __init__(self,args,embeddings):
        super(BiRNN, self).__init__()
        self.hidden_size = args['hidden_size']
        self.batch_size = args['batch_size']
        self.drop_embed=args['drop_embed']
        self.drop_fc=args['drop_fc']
        self.embedsize=args["embed_size"]
        self.drop_hidden=args['drop_hidden']
        self.seq_model_name=args["seq_model"]
        self.weights =args["weights"]
        self.embedding = nn.Embedding(args["vocab_size"], self.embedsize)
        self.embedding.weight = nn.Parameter(torch.tensor(embeddings.astype(np.float32), dtype=torch.float32))
        self.embedding.weight.requires_grad = args["train_embed"]
        if(args["seq_model"]=="lstm"):
            self.seq_model = nn.LSTM(args["embed_size"], self.hidden_size, bidirectional=True, batch_first=True,dropout=self.drop_hidden)
        elif(args["seq_model"]=="gru"):
            self.seq_model = nn.GRU(args["embed_size"], self.hidden_size, bidirectional=True, batch_first=True,dropout=self.drop_hidden) 
        self.linear1 = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, args['num_classes'])
        self.dropout_embed = nn.Dropout2d(self.drop_embed)
        self.dropout_fc = nn.Dropout(self.drop_fc)
        self.num_labels=args['num_classes']
        
        
        
    def forward(self,input_ids=None,attention_mask=None,attention_vals=None,labels=None,device=None):
        batch_size=input_ids.size(0)
        h_embedding = self.embedding(input_ids)
        h_embedding = torch.squeeze(self.dropout_embed(torch.unsqueeze(h_embedding, 0))).view(batch_size,input_ids.shape[1],self.embedsize)
        if(self.seq_model_name=="lstm"):
            _, hidden = self.seq_model(h_embedding)
            hidden=hidden[0]
        else:
            _, hidden = self.seq_model(h_embedding)
            
        if(debug):
            print(hidden.shape)
        hidden = hidden.transpose(0, 1).contiguous().view(batch_size, -1) 
        hidden = self.dropout_fc(hidden)
        hidden = torch.relu(self.linear1(hidden))  #batch x hidden_size
        hidden = self.dropout_fc(hidden)
        logits = self.linear2(hidden)
        if labels is not None:
            loss_funct = torch.nn.CrossEntropyLoss(weight=torch.tensor(self.weights).to(device),reduction='mean')
            loss_logits =  loss_funct(logits.view(-1, self.num_labels), labels.view(-1)) 
            return (loss_logits,logits)  
        return (logits,)
    
    
    
    def init_hidden(self, batch_size):
        return cuda_available(torch.zeros(2, self.batch_size, self.hidden_size))


class LSTM_bad(BiRNN):
    def __init__(self,args):
        super().__init__(args)
        self.seq_model = nn.LSTM(args["embed_size"], self.hidden_size, bidirectional=False, batch_first=True,dropout=self.drop_hidden)
    def forward(self,x,x_mask):
        batch_size=x.size(0)
        h_embedding = self.embedding(x)
        h_embedding = torch.squeeze(self.dropout_embed(torch.unsqueeze(h_embedding, 0))).view(batch_size,x.shape[1],self.embedsize)
        _, hidden = self.seq_model(h_embedding)
        hidden=hidden[0]
        if(debug):
            print(hidden.shape)
        hidden = hidden.transpose(0, 1).contiguous().view(batch_size, -1) 
        hidden = self.dropout_fc(hidden)
        return (self.linear2(hidden))  
       
    
    

class CNN_GRU(BiRNN):
    def __init__(self,args,embeddings):
        super().__init__(args,embeddings)
        self.conv1 = nn.Conv1d(self.embedsize,100, 2)
        self.conv2 = nn.Conv1d(self.embedsize,100, 3,padding=1)
        self.conv3 = nn.Conv1d(self.embedsize,100, 4,padding=2)
        self.maxpool1D = nn.MaxPool1d(4, stride=4)
        self.seq_model = nn.GRU(100, 100, bidirectional=False, batch_first=True,dropout=self.drop_hidden)
        self.out = nn.Linear(100, args["num_classes"])
        
    def forward(self,input_ids=None,attention_mask=None,attention_vals=None,labels=None,device=None):
        batch_size=input_ids.size(0)
        h_embedding = self.embedding(input_ids)
        h_embedding = self.dropout_embed(h_embedding)
        new_conv1=self.maxpool1D(self.conv1(h_embedding.permute(0,2,1)))
        new_conv2=self.maxpool1D(self.conv2(h_embedding.permute(0,2,1)))
        new_conv3=self.maxpool1D(self.conv3(h_embedding.permute(0,2,1)))
        concat=self.maxpool1D(torch.cat([new_conv1, new_conv2,new_conv3], dim=2))
        h_seq, _ = self.seq_model(concat.permute(0,2,1))
        global_h_seq=torch.squeeze(global_max_pooling(h_seq, 1, 1)) 
        global_h_seq = self.dropout_fc(global_h_seq)
        output=self.out(global_h_seq)
        if labels is not None:
            loss_funct = torch.nn.CrossEntropyLoss(weight=torch.tensor(self.weights).to(device),reduction='mean')
            loss_logits =  loss_funct(output.view(-1, self.num_labels), labels.view(-1)) 
            return (loss_logits,output)  
        return (output,)
    
        return output
    
class BiAtt_RNN(BiRNN):
    def __init__(self,args,embeddings,return_att):
        super().__init__(args,embeddings)
        if(args['attention']=='sigmoid'):
             self.seq_attention = Attention_LBSA_sigmoid(self.hidden_size * 2, args['max_length'])
        else:
             self.seq_attention = Attention_LBSA(self.hidden_size * 2, args['max_length'])
        self.linear = nn.Linear(self.hidden_size * 2, args["batch_size"])
        self.relu = nn.ReLU()
        self.out = nn.Linear(args["batch_size"], args["num_classes"])
        self.return_att=False
        self.lam=args['att_lambda']
        self.train_att =args['train_att']
        
        
    def forward(self, input_ids=None,attention_mask=None,attention_vals=None,labels=None,device=None):
        h_embedding = self.embedding(input_ids)
        h_embedding = torch.squeeze(self.dropout_embed(torch.unsqueeze(h_embedding, 0))).view(input_ids.shape[0],input_ids.shape[1],self.embedsize)

        h_seq, _ = self.seq_model(h_embedding)
        if(debug):
            print("output",h_seq.shape)

        h_seq_atten,att = self.seq_attention(h_seq,attention_mask)
        if(debug):
            print("h_seq_atten",h_seq_atten.shape)

        conc=h_seq_atten
        conc=self.dropout_fc(conc)
        conc = self.relu(self.linear(conc))
        conc = self.dropout_fc(conc)
        outputs = self.out(conc)
        outputs=(outputs,)
        
        if labels is not None:
            loss_funct = torch.nn.CrossEntropyLoss(weight=torch.tensor(self.weights).to(device),reduction='mean')
            loss_logits =  loss_funct(outputs[0].view(-1, self.num_labels), labels.view(-1))
            loss= loss_logits
           
            if(self.train_att):
                loss_atts = self.lam*masked_cross_entropy(att,attention_vals,attention_mask)
                loss = loss+loss_atts
            outputs = (loss,) + outputs
       
       
        outputs= outputs+(att,)
        return outputs
        


if __name__ == '__main__':
    args_dict = {
        "batch_size":10,
        "hidden_size":256,
        "epochs":10,
        "embed_size":300,
        "drop":0.1,
        "learning_rate":0.001,
        "vocab_size":10000,
        "num_classes":3,
        "embeddings":np.array([]),
        "seq_model":"lstm",
        "drop_embed":0.1,
        "drop_fc":0.1,
        "drop_hidden":0.1,
        "train_embed":False
        
        }
#     BiRNN(args_dict)
#     BiAtt_RNN(args_dict)
#     BiSCRAT_RNN(args_dict)
    CNN_GRU(args_dict)
