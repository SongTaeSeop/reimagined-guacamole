#%%
from transformers import TFBertForMaskedLM
from transformers import AutoTokenizer
# %%
# import torch, gc
# gc.collect()
# torch.cuda.empty_cache()
model = TFBertForMaskedLM.from_pretrained('klue/bert-base', from_pt=True)
tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
# %%
inputs = tokenizer('축구는 정말 재미있는 [MASK]다.', return_tensors='tf')
#%%
print(inputs['token_type_ids'])
print(inputs['input_ids'])
print(inputs['attention_mask'])
# %%
from transformers import FillMaskPipeline
pip = FillMaskPipeline(model=model, tokenizer=tokenizer)
pip('축구는 정말 재미있는 [MASK]다.')

#%%
pip('어벤져스는 정말 재미있는 [MASK]다.')
# %%
