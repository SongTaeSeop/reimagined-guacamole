#%%
import pandas as pd
!pip install transformers
from transformers import BertTokenizer
#%%
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") # Bert-base tokenizer
result = tokenizer.tokenize('Here is the sentence I want embeddings for.')
print(result)
# %%
