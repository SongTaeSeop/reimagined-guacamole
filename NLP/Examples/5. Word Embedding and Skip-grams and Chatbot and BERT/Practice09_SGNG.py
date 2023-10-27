#%% 
import pandas as pd
import numpy as np

from sklearn.datasets import fetch_20newsgroups

#%% ##################################################################################
# Dataset loading
dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
dataset = dataset.data
print(f"Sample data: {dataset[0]}")
print(f"Total number of data : {len(dataset)}")
news_df = pd.DataFrame({'document':dataset})
news_df

# %% ##################################################################################
# Preprocessing

# 결측값 제거
news_df.replace("", float("NaN"), inplace=True)
processed_news_df = news_df.dropna().reset_index(drop=True)

# 중복값 제거
processed_news_df = processed_news_df.drop_duplicates(['document']).reset_index(drop=True)

# 특수문자 제거
processed_news_df['document'] = processed_news_df['document'].str.replace("[^a-zA-Z]", " ")

# 단어 길이가 2이하인 단어 제거
processed_news_df['document'] = processed_news_df['document'].apply(lambda x: ' '.join([token for token in x.split() if len(token) > 2]))

# 전체 길이가 200 이하이거나 전체 단어수가 5개 이하인 문서 제거
processed_news_df = processed_news_df[processed_news_df.document.apply(lambda x: len(str(x)) <= 200 and len(str(x).split()) > 5)].reset_index(drop=True)
# 정규화 (대소문자 통일)
processed_news_df['document'] = processed_news_df['document'].apply(lambda x: x.lower())

print(f"Total number of the filtered data : {len(processed_news_df)}")
print(f"Sample data after preprocessing : {processed_news_df.iloc[0][0]}")

# %% 
# 불용어 제거 (패키지 활용) + 띄어쓰기 단위로 문장 분리

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

stop_words = stopwords.words('english')

tokenized_doc = processed_news_df['document'].apply(lambda x: x.split())
tokenized_doc = tokenized_doc.apply(lambda x: [s_word for s_word in x if s_word not in stop_words])

tokenized_doc
# %% ##################################################################################
# Tokenization
# 단어 수가 1개 이하인 샘플을 제거
drop_train = [index for index, sentence in enumerate(tokenized_doc) if len(sentence) <= 1]
tokenized_doc = np.delete(tokenized_doc, drop_train, axis=0)

print(f"Total number of training data : {len(tokenized_doc)}")

# %% 
# 정수 인코딩 진행
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
tokenizer.fit_on_texts(tokenized_doc)

word2idx = tokenizer.word_index
idx2word = {value : key for key, value in word2idx.items()}
encoded = tokenizer.texts_to_sequences(tokenized_doc)

vocab_size = len(word2idx) + 1 
print(f"Size of vocabulary : {vocab_size}")
print(encoded[0])
# %% ##################################################################################
# SGNG
from tensorflow.keras.preprocessing.sequence import skipgrams

# Negative sampling
skip_grams = [skipgrams(sample, vocabulary_size=vocab_size, window_size=10) for sample in encoded[:5]]
print(f"Total number of samples : {len(skip_grams)}")

# Check the immediate result
pairs, labels = skip_grams[0][0], skip_grams[0][1]
print(f"first 3 pairs: {pairs[:3]}")
print(f"first 3 labels: {labels[:3]}")

#%% Assign the labels for Positive sample -> 1, Negative sample -> 0
for i in range(5):
  print("({:s} ({:d}), {:s} ({:d})) -> {:d}".format(
    idx2word[pairs[i][0]], pairs[i][0], 
    idx2word[pairs[i][1]], pairs[i][1], 
    labels[i])
  )

# %% Split data for training
# Number of training dataset: 1000
training_dataset = [skipgrams(sample, vocabulary_size=vocab_size, window_size=10) for sample in encoded[:1000]]

#%% ##################################################################################
# Word embedding 구축
# !pip install pydot
# (optional) sudo apt install graphviz (in terminal)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, Reshape, Activation, Input
from tensorflow.keras.layers import Dot
from tensorflow.keras.utils import plot_model
from IPython.display import SVG

# Dim. of embedding vector = 100
# Add two layers
embedding_dim = 100

w_inputs = Input(shape=(1, ), dtype='int32') # for target word
word_embedding = Embedding(vocab_size, embedding_dim)(w_inputs)

c_inputs = Input(shape=(1, ), dtype='int32') # for context word
context_embedding  = Embedding(vocab_size, embedding_dim)(c_inputs)

# %% Modeling
dot_product = Dot(axes=2)([word_embedding, context_embedding])
dot_product = Reshape((1,), input_shape=(1, 1))(dot_product)
output = Activation('sigmoid')(dot_product)

model = Model(inputs=[w_inputs, c_inputs], outputs=output)
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam')
plot_model(model, to_file='model3.png', show_shapes=True, show_layer_names=True, rankdir='TB')

# %% Training (20 epochs)
for epoch in range(20):
  loss = 0
  for _, elem in enumerate(skip_grams):
    first_elem = np.array(list(zip(*elem[0]))[0], dtype='int32')
    second_elem = np.array(list(zip(*elem[0]))[1], dtype='int32')
    labels = np.array(elem[1], dtype='int32')
    X = [first_elem, second_elem]
    Y = labels
    loss += model.train_on_batch(X,Y)  
  print('Epoch :',epoch + 1, 'Loss :',loss)

# %% ##################################################################################
# Embedding 품질 확인 (gensim 활용)
import gensim

with open('vectors.txt' ,'w') as f: # save the vectors
    f.write('{} {}\n'.format(vocab_size-1, embedding_dim))
    vectors = model.get_weights()[0]
    for word, i in tokenizer.word_index.items():
        f.write('{} {}\n'.format(word, ' '.join(map(str, list(vectors[i, :])))))

# %% Load model
# gensim.models.KeyedVectors.load_word2vec_forma() ==> 단어 벡터간 유사도 계산
w2v = gensim.models.KeyedVectors.load_word2vec_format('./vectors.txt', binary=False)
# %% Similarity test
print(w2v.most_similar(positive=['apple']))
print(w2v.most_similar(positive=['doctor']))
print(w2v.most_similar(positive=['engine']))
# %%
