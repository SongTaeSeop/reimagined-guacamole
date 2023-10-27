#%%
import tensorflow as tf
from transformers import TFBertForNextSentencePrediction
from transformers import AutoTokenizer
# %%
model = TFBertForNextSentencePrediction.from_pretrained('bert-base-uncased')
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
#%%
prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
next_sentence = "pizza is eaten with the use of a knife and fork. In casual settings, however, it is cut into wedges to be eaten while held in the hand."
# %%
encoding = tokenizer(prompt, next_sentence, return_tensors='tf')
print(encoding['input_ids'])
print(tokenizer.decode(encoding['input_ids'][0]))
# %%
logits = model(encoding['input_ids'], token_type_ids=encoding['token_type_ids'])[0]
softmax = tf.keras.layers.Softmax()
probs = softmax(logits)
print(probs)
print('최종 예측 레이블 :', tf.math.argmax(probs, axis=-1).numpy())
#%%
# 상관없는 두 개의 문장
prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
next_sentence = "The sky is blue due to the shorter wavelength of blue light."
encoding = tokenizer(prompt, next_sentence, return_tensors='tf')

logits = model(encoding['input_ids'], token_type_ids=encoding['token_type_ids'])[0]

softmax = tf.keras.layers.Softmax()
probs = softmax(logits)
print('최종 예측 레이블 :', tf.math.argmax(probs, axis=-1).numpy())