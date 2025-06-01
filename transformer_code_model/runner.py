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
