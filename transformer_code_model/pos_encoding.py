import torch.nn as nn 

class sequence_encoding(nn.Module): 

    def __init__(self,d_model = 128): 
        super().__init__()
        self.d_model = 128 

    def __call__(self,pos_len):
        pos_encoding = torch.randn(pos_len,self.d_model)
        
        # we are use sin , cos ( odd, even ) function  for diff - diff values 
        # the thing we are need diff, continuous , we are get diff values with diff pos, d_model(i) 
        # so we are use sin, cos  ( go extremly deep that important) 
        '''
        the question is this why ? 
        1. sin and cos 
        2. why sin--> even  cos --> odd index 
        3. why? 
        
        '''
