#%% ##################################################################################
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

# %%
target = "화산의 검은 매화를 흉내 내지 않는다. \
화산의 검은 매화를 피워낸다. 매화가 아니라 개화이다."

tokenizer = Tokenizer()
tokenizer.fit_on_texts([target])

for key, idx in tokenizer.word_index.items():
    print(f"{key} : {idx}")

# %% One-hot encoding 구현
encoded = tokenizer.texts_to_sequences([target])[0]
print(encoded)

onehot_encoded = tf.keras.utils.to_categorical(encoded)
print(onehot_encoded)

# %% ##################################################################################
# Bag-of-words 구현
document_1 = "화산의 검은 매화를 흉내 내지 않는다. \
화산의 검은 매화를 피워낸다. 매화가 아니라 개화이다."

document_2 = "매화의 화려함에 눈을 빼앗기고, \
    검의 날카로움에 영혼이 홀린 이들은 결코 화산 검학의 진수에 도달할 수 없다."
    
document_3 = "저에게 있어 화산은 그저 화산입니다."
#%%
from sklearn.feature_extraction.text import CountVectorizer

training_documents = [document_1, document_2, document_3]

bow_vectorizer = CountVectorizer()
bow_vectorizer.fit(training_documents)

word_index = bow_vectorizer.vocabulary_

for key, idx in sorted(word_index.items()):
    print(f"({key}: {idx})")
# %%
bow_vector_1 = bow_vectorizer.transform([document_1])
bow_vector_2 = bow_vectorizer.transform([document_2])
bow_vector_3 = bow_vectorizer.transform([document_3])

print(bow_vector_1.toarray())
print(bow_vector_2.toarray())
print(bow_vector_3.toarray())
# %%
import pandas as pd

result = []
vocab = list(word_index.keys())

for i in range(0, len(training_documents)):
    result.append([])
    d = training_documents[i]
    for j in range(0, len(vocab)):
        target = vocab[j]
        result[-1].append(d.count(target))
        
tf_ = pd.DataFrame(result, columns = vocab)
tf_

# %% ##################################################################################
# TF-IDF using sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

tfidfv = TfidfVectorizer().fit(training_documents)
sk_tf_idf = tfidfv.transform(training_documents).toarray()
print(sk_tf_idf)
print(tfidfv.vocabulary_)

# %%
from sklearn.metrics.pairwise import euclidean_distances
# %%
def jaccard_similarity(doc1, doc2):
    s1 = set(doc1)
    s2 = set(doc2)
    
    return float(len(s1.intersection(s2)) / len(s1.union(s2)))