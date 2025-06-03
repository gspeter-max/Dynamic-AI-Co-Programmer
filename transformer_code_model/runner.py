#! pip install transformers
# from transformers.models.gpt2 import GPT2Tokenizer
import requests
import torch
import math 
import torch.nn as nn 


# hitting_url = 'https://raw.githubusercontent.com/gspeter-max/cv_project/main/ctr_torch.py'
# response = requests.get(url = hitting_url)
# github_code = response.text

# with open('transformer_data.text','w') as f :
#     f.write(github_code)


# tok = GPT2Tokenizer.from_pretrained('openai-community/gpt2')

# vocab_size = len(tok.get_vocab())
# text_embedding = torch.nn.Embedding(vocab_size,128)
# text_sequence = torch.tensor(tok(github_code)['input_ids'])
# sequenced_embedding = text_embedding(text_sequence)


vocab = {"<pad>": 0, "The": 1, "cat": 2, "sat": 3, "on": 4, "the": 5, "mat": 6, "dog": 7, "ran": 8}
vocab_size = len(vocab)

# Example sentence (list of word IDs)
sentence_ids_1 = [vocab["The"], vocab["cat"], vocab["sat"], vocab["on"], vocab["the"], vocab["mat"]]
sentence_ids_2 = [vocab["The"], vocab["dog"], vocab["ran"], vocab["on"], vocab["the"], vocab["mat"]]

# Create a batch of token IDs (required for NN modules)
# Shape: (batch_size, seq_len)
input_batch_ids = torch.tensor([sentence_ids_1, sentence_ids_2]) 
print(input_batch_ids.shape)
batch_dims , seq_len = input_batch_ids.shape 

class MultiheadAttention(nn.Module): 

    def __init__(self,vocab_size = vocab_size,d_model = 128,masked = True, num_head = 4,position_encoding_max_len = 10): 
        
        super().__init__() 

        self.d_model = d_model 
        self.num_head = num_head 
        self.is_causal_mask = masked 

        assert self.d_model % self.num_head == 0 , RuntimeError(f'num_head is not compatable with d_model : {self.d_model / self.num_head} ')
        self.head_dim = self.d_model // self.num_head 

        self.embedding = nn.Embedding(vocab_size, self.d_model )

        self.w_q = nn.Linear(self.d_model,self.d_model, bias= False)
        self.w_k = nn.Linear(self.d_model,self.d_model,bias = False)
        self.w_v = nn.Linear(self.d_model,self.d_model, bias = False)

        self.w_o = nn.Linear(self.d_model, self.d_model,bias = False)

        self.register_buffer('Positional_Encoding',self._get_position_encoding(position_encoding_max_len).unsqueeze(0))
        
    
    def _get_position_encoding(self,max_len): 
        
        PE = torch.zeros(max_len , self.d_model)

        i = torch.arange(int(self.d_model/2))
        pos = torch.arange(max_len).unsqueeze(1)

        angle = pos / 10000 **((2 * i)/self.d_model)

        PE[:,::2] = torch.sin(angle)
        PE[:,1::2] = torch.cos(angle)

        return PE 

    def forward(self,input_seq): 
        
        embedding = self.embedding(input_seq)
        Batch, seq_len,embedding_dim  = embedding.shape
        print(f'embedding shape : {embedding.shape}')

        Position_Encoding = self.Positional_Encoding[:,:seq_len,:]
        embedding = embedding + Position_Encoding

        q = self.w_q(embedding)
        k = self.w_k(embedding)
        v = self.w_v(embedding)


        q = q.view( Batch, self.num_head, -1 , self.head_dim )
        k = k.view( Batch,self.num_head, -1 , self.head_dim )
        v = v.view( Batch ,self.num_head, -1 , self.head_dim )

        raw_score = (q @ k.transpose(-2,-1)) / math.sqrt(self.head_dim)
        
        if self.is_causal_mask: 
            print('Applying Masking') 
            masked = torch.tril(torch.ones(raw_score.shape, device=raw_score.device), diagonal= 0)
            masked = masked.masked_fill(masked == 0, float('-inf'))
            
            raw_score = torch.abs(raw_score) * masked   
        
        attention_score = torch.softmax(raw_score, dim=-1)  
        
        temp = attention_score @ v
        temp = temp.view(Batch ,seq_len, -1)
        return self.w_o(temp)

MHA = MultiheadAttention()
result = MHA.__call__(input_batch_ids)

#! pip install transformers
from transformers.models.gpt2 import GPT2Tokenizer
import requests
import torch


hitting_url = 'https://raw.githubusercontent.com/gspeter-max/cv_project/main/ctr_torch.py'
response = requests.get(url = hitting_url)
github_code = response.text

with open('transformer_data.text','w') as f :
    f.write(github_code)


tok = GPT2Tokenizer.from_pretrained('openai-community/gpt2')

vocab_size = len(tok.get_vocab())
text_embedding = torch.nn.Embedding(vocab_size,128)
text_sequence = torch.tensor(tok(github_code)['input_ids'])
sequenced_embedding = text_embedding(text_sequence)


import torch

class position_encoding:

    def __init__(self,d_model = 128,pos_len = 32):
        self.d_model = d_model
        self.pos_len = pos_len

    def __call__(self):

        PE = torch.randn(self.pos_len,self.d_model)
        pos = torch.arange(self.pos_len).view(-1, 1)                # for brodcasting we are make 2D
        i = torch.arange(0,int(self.d_model/2))                     # for brodcasting we are make 2D

        w_i = 10000**(i/ self.d_model)                              # I AM TRING TO MAKE  MODEL FOR SHORT CODE GENRATION SO NOT NEED TOO LONG TERM PATTERNS

        '''
        $ 2i when you working with you do not care about very tini information (medium range (short, long) ))
        means i --> very small patterns like ( def function ()) also for "def" also for "()" )  frequency --> high , and decrease repidly for log term

        # 2i --> medium range paterns decent patterns like "def" ")" and also for long term and best in log term compared to "i" and no high frequency but
            higher then "i"
            like
            i --> [0.3, 0.8, 0.01]
            2i --> [ 0.6, 1.6,0.02] # you see things are double and make more empect in log term
            4i --> 4[i] # THAT INCREASE THE WAVELENGTH

        '''

        angle = pos / w_i

        PE[:,::2] = torch.sin( angle )
        PE[:,1::2] = torch.cos( angle )

        return PE
import torch
import torch.nn as nn
import torch.nn.functional as F

class Feed_Forward_Network(nn.Module):

    def __init__(self,input_dims):
        self.layer1 = nn.Linear(input_dims, 3*input_dims)
        self.layer2 = nn.Linear(3*input_dims, input_dims)

    def forward(self,x):
        return self.layer2(F.relu(self.layer1(x)))

class Encoder(nn.Module):

    def __init__(self,d_model = 128):
        super().__init__()
        self.Sentance = S

        self.encoder  = {value : index for index, value in enumerate(S)}
        self.encoded_value = torch.tensor([self.encoder.get(value) for value in S])
        self.d_model = d_model
        self.embedding_ = nn.Embedding(len(S),d_model)

        self.w_q = nn.Linear(d_model,d_model)
        self.w_k = nn.Linear(d_model,d_model)
        self.w_v = nn.Linear(d_model,d_model)


    def softmax(self,x):

        x = x - torch.max(x,dim = 1,keepdim = True).values
        exp_x = torch.exp(x)

        return exp_x / torch.sum(exp_x, dim = 1, keepdim = True)

    def position_encoding(self):

        PE = torch.zeros(len(self.Sentance),self.d_model)
        pos = torch.arange(len(self.Sentance)).view(-1,1)
        i = torch.arange(128/2)

        power = 2*i/self.d_model
        angle = pos/ 10000**(power)

        PE[:,::2]  = torch.sin(angle)
        PE[:,1::2] = torch.cos(angle)

        return PE

    def forward(self):

        embedding = self.embedding_(self.encoded_value)
        '''
        because attention solve the problem  ( speed and parallel computation ) with gpu doing parallel processing
        parallel --> node or tokens break over sequence so maintaining the sequence we are do add information about sequence
        we are use position encoding

        that time model also know genius peter != peter genius  have diff

        and that position information we are add to embedding so attention learn from embedding and positions  and combine the embedding ( attention process )
        and return next embedding uisng token and that sequences
        more tell me about that

        '''
        PE = self.position_encoding()
        embedding = embedding + PE

        q,k,v  = self.w_q(embedding), self.w_k(embedding) , self.w_v(embedding)

        raw_score = (q @ k.T) / self.d_model**0.5
        attention_weight = self.softmax(raw_score)

        return attention_weight ,attention_weight @ v

import math
import torch
import torch.nn as nn
import warnings

class MultiheadAttention(nn.Module):

    def __init__(self,d_model = 128, num_head = 4,masked = False):

        super().__init__()

        self.d_model = d_model
        self.num_head = num_head
        self.masked = masked

        assert self.d_model % self.num_head == 0 , RuntimeError(f'num_head is not compatable with d_model : {self.d_model / self.num_head} ')
        self.head_dim = self.d_model // self.num_head

        self.w_q = nn.Linear(self.d_model,self.d_model, bias= False)
        self.w_k = nn.Linear(self.d_model,self.d_model,bias = False)
        self.w_v = nn.Linear(self.d_model,self.d_model, bias = False)

        self.w_o = nn.Linear(self.d_model, self.d_model,bias = False)

    def forward(self,embedding):

        Batch, seq_len,_  = embedding.shape

        q = self.w_q(embedding)
        k = self.w_k(embedding)
        v = self.w_v(embedding)


        q = q.view( Batch, self.num_head, seq_len , self.head_dim )
        k = k.view( Batch,self.num_head, seq_len , self.head_dim )
        v = v.view( Batch ,self.num_head, seq_len , self.head_dim )

        raw_score = (q @ k.transpose(-2,-1)) / math.sqrt(self.head_dim)

        if self.masked is not  None :

            mask = torch.triu(torch.ones(seq_len, self.head_dim), diagonal = 1 ).unsqueeze(1).unsqueeze(2)
            mask = mask.masked_fill(mask == 1, float('-inf'))
            raw_score = raw_score + mask

        attention_score = torch.softmax(raw_score, dim = -1)
        temp = attention_score @ v

        temp = temp.view(Batch ,seq_len, self.num_head * self.head_dim)
        return self.w_o(temp)

class corss_attention(nn.Module):

    def __init__(self, d_model = 128, num_head = 4):
        super().__init__()

        self.d_model = d_model
        self.num_head = num_head

        assert self.d_model % self.num_head == 0 , RuntimeError(f'num_head is not compatable with d_model : {self.d_model / self.num_head} ')
        self.head_dim = self.d_model // self.num_head

        self.w_q = nn.Linear(self.d_model,self.d_model, bias= False)
        self.w_k = nn.Linear(self.d_model,self.d_model,bias = False)
        self.w_v = nn.Linear(self.d_model,self.d_model, bias = False)

        self.w_o = nn.Linear(self.d_model, self.d_model,bias = False)


    def forward(self,decoder_output , encoder_output , padding_mask = None):

        Batch, target_len,_  = decoder_output.shape
        _, sql_len, _ = encoder_output.shape

        q = self.w_q(decoder_output)
        k = self.w_k(encoder_output)
        v = self.w_v(encoder_output)


        q = q.view( Batch, self.num_head, target_len, self.head_dim )
        k = k.view( Batch,self.num_head, sql_len , self.head_dim )
        v = v.view( Batch ,self.num_head, sql_len , self.head_dim )

        raw_score = (q @ k.transpose(-2,-1)) / math.sqrt(self.head_dim)
        if padding_mask is not None:

            masked = padding_mask.masked_fill(padding_mask == 0, float('-inf'))
            masked = masked.unsqueeze(1).unsqueeze(2)

            raw_score = raw_score + masked
        else :

            warnings.warn('that is None ', RuntimeWarning)

        cross_attention = torch.softmax(raw_score, dim=-1)

        temp = cross_attention  @ v
        temp = temp.view(Batch ,target_len , -1)
        return self.w_o(temp)


