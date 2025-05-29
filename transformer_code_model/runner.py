
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

