# #%% 
import tensorflow
from tensorflow.keras.layers import Embedding

embedding_layer = Embedding(1000, 64)

#%% ##################################################################################
# IMDB 영화 리뷰 감성 예측
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

from tensorflow.keras.datasets import imdb
from tensorflow.keras import preprocessing

max_features = 10000
maxlen = 20

# 데이터셋을 리스트 형태로 로드
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# 리스트를 (samples, maxlen)크기의 2D 정수텐서로 변환
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Embedding
import numpy as np

model = Sequential()

# Embedding layer의 출력 크기 (samples, maxlen, 8)
model.add(Embedding(10000, 8, input_length=maxlen))

# 3D embedding tensor를 (samples, maxlen * 8)의 크기의 2D tensor로 flatten
model.add(Flatten())

# classifier
model.add(Dense(1, activation = 'sigmoid'))
model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary()

# %%
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# %%
# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print("Loss:", loss)
print("Accuracy:", accuracy)

# %%
predictions = model.predict(x_test)

predicted_labels = [1 if prediction > 0.5 else 0 for prediction in predictions]

word_index = imdb.get_word_index()
reverse_word_index = {index: word for word, index in word_index.items()}

for i in range(len(x_test)):
    predicted_sentence = ' '.join(reverse_word_index.get(index - 3, '?') for index in x_test[i])

    print("Predicted Sentence:", predicted_sentence)
    print("Predicted Label:", predicted_labels[i])
    print()
    
# %% ##################################################################################
# Word2Vec 구현
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import nltk
nltk.download('punkt')

sentences = [
    "I enjoy playing football",
    "Football is a popular sport",
    "I love watching football matches"
]

tokenized_sentences = [nltk.word_tokenize(sentence.lower()) for sentence in sentences]

# CBOW model
model = Word2Vec(tokenized_sentences, vector_size=100, window=2, min_count=1, sg=0)

class LossLogger(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0
        self.losses = []
        
    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        self.losses.append(loss)
        print(f"Loss after epoch {self.epoch}: {loss}")
        self.epoch += 1

loss_logger = LossLogger()

model.train(tokenized_sentences, total_examples=len(tokenized_sentences), epochs=10, callbacks=[loss_logger])

word_vector = model.wv["football"]
print(f"Word vector for 'football': {word_vector}")

# %%
