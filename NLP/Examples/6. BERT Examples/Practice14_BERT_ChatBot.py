#%%
import urllib.request
import pandas as pd

#%% Data download
urllib.request.urlretrieve("https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv", filename="ChatBotData.csv")
#%% Data loading
train_dataset = pd.read_csv('ChatBotData.csv')

print(f"전체 데이터셋 개수: {len(train_dataset)}")
# train_dataset

#%% Data preprocessing

# 결측값 확인
train_dataset.replace("", float("NaN"), inplace=True)
print(train_dataset.isnull().values.any())

# 중복 제거
# Question: 열을 기준으로 중복제거
train_dataset = train_dataset.drop_duplicates(['Q']).reset_index(drop=True)
print(f"필터링된 데이터셋 총 개수 : {len(train_dataset)}")

# Answer: 열을 기준으로 중복제거
train_dataset = train_dataset.drop_duplicates(['A']).reset_index(drop=True)
print(f"필터링된 데이터셋 총 개수 : {len(train_dataset)}")
# train_dataset

#%% Data distribution check

from matplotlib import pyplot as plt

question_list = list(train_dataset['Q'])
answer_list = list(train_dataset['A'])

print('질문의 최대 길이 :',max(len(question) for question in question_list))
print('질문의 평균 길이 :',sum(map(len, question_list))/len(question_list))
plt.hist([len(question) for question in question_list], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()
# %%
print('답변의 최대 길이 :',max(len(answer) for answer in answer_list))
print('답변의 평균 길이 :',sum(map(len, answer_list))/len(answer_list))
plt.hist([len(answer) for answer in answer_list], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

#%% 답변 후보 목록 구성

import random

print(f"question 개수: {len(question_list)}")
print(f"answer 개수: {len(answer_list)}")

response_candidates = random.sample(answer_list, 100)

response_candidates[:10]
# %% KoBERT-Transformers loading

!pip install kobert-transformers
#%%
import torch
from kobert_transformers import get_kobert_model, get_distilkobert_model
# %%
model = get_kobert_model()
model.eval()

input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
attention_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
output = model(input_ids, attention_mask, token_type_ids)
output

#%%
from kobert_transformers import get_tokenizer

tokenizer = get_tokenizer()
# %%
tokenizer.tokenize("[CLS] 한국어 모델을 공유합니다. [SEP]")
#%%
tokenizer.convert_tokens_to_ids(['[CLS]', '▁한국', '어', '▁모델', '을', '▁공유', '합니다', '.', '[SEP]'])
# %% Ranking model
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

#%% Answer ranking pipeline
def get_cls_token(sentence):
    model.eval()
    tokenized_sent = tokenizer(
      sentence,
      return_tensors="pt",
      truncation=True,
      add_special_tokens=True,
      max_length=128
    )
    input_ids = tokenized_sent['input_ids']
    attention_mask = tokenized_sent['attention_mask']
    token_type_ids = tokenized_sent['token_type_ids']

    with torch.no_grad():
        output = model(input_ids, attention_mask, token_type_ids)

    cls_output = output[1]
    cls_token = cls_output.detach().cpu().numpy()

    return cls_token

def predict(query, candidates):
  candidates_cls = []
  
  for cand in candidates:
    cand_cls = get_cls_token(cand)
    candidates_cls.append(cand_cls)
  
  candidates_cls = np.array(candidates_cls).squeeze(axis=1)

  queury_cls = get_cls_token(query)
  similarity_list = cosine_similarity(queury_cls, candidates_cls)

  target_idx = np.argmax(similarity_list)

  return candidates[target_idx]

# %% get_cls_token() test
query = '너 요즘 바빠?'
query_cls_hidden = get_cls_token(query)
print(query_cls_hidden)
print(query_cls_hidden.shape)

# %% prediction test
sample_query = '너 요즘 바빠?'
sample_candidates = ['아니 별로 안바빠','바쁘면 바보','사자와 호랑이가 싸우면 누가 이길까', "내일은 과연 해가 뜰까"]

predicted_answer = predict(query, sample_candidates)

print(f"predicted_answer = {predicted_answer}")

#%% answer evaluation
user_query = '너 요즘 바빠?'
predicted_answer = predict(query, response_candidates)
print(f"predicted_answer = {predicted_answer}")
# %%
response_candidates = random.sample(answer_list, 100)
user_query = '나 요즘 너무 힘들어'
predicted_answer = predict(query, response_candidates)
print(f"predicted_answer = {predicted_answer}")
# %%
end = 1
while end == 1 :
    sentence = input("하고싶은 말을 입력해주세요 : ")
    if len(sentence) == 0 :
        break
    predicted_answer = predict(sentence, response_candidates)
    print(f"Q: {sentence}")
    print(f"A: {predicted_answer}")
    print("\n")
# %%
