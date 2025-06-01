import torch 
import torch.nn as nn 

a = torch.randn(4, 4) 
masked = torch.tril(torch.ones(4, 4), diagonal=1) 
# masked = masked.masked_fill(masked == 0, float('-inf'))   
# print(torch.abs(a)  * masked) 
masked1 = torch.tril(torch.ones(4, 4), diagonal=0) 

masked2 = torch.triu(torch.ones(4, 4), diagonal=1) 
# masked = masked.masked_fill(masked == 0, float('-inf'))   
# print(torch.abs(a)  * masked) 
masked3 = torch.triu(torch.ones(4, 4), diagonal=0) 

print(masked)
print(masked1) 
print(masked2) 
print(masked3)
