import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 설명: Note > 자연어 처리 > 5-1. 코사인 유사도(Cosine Similarity)
data = pd.read_csv("movies_metadata.csv", low_memory = False)
[print(i,"번째 열:", j) for i, j in enumerate(data.columns)]

data = data.head(20000)

before = data["overview"].isnull().sum()
data["overview"] = data["overview"].fillna('')
after = data["overview"].isnull().sum()
print("overview의 결측값의 수 (before):", before)
print("overview의 결측값의 수 (after):", after)

tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(data["overview"])
print("TF-IDF의 크기(shape):", tfidf_matrix.shape)

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
print("코사인 유사도 결과:", cosine_sim.shape)

title_to_index = dict(zip(data["title"], data.index))

idx = title_to_index["Father of the Bride Part II"]
print(idx)

def get_recommendations(title, cosine_sim = cosine_sim):
	# 주어진 영화 타이틀의 index에 해당하는 열의 코사인 유사도를 sim_scores 리스트에 저장한다.
	# sim_scores: [인덱스, 점수]
	idx = title_to_index[title]
	sim_scores = list(enumerate(cosine_sim[idx]))

	# sim_scores[1]->(점수)순으로 정렬한다.
	sim_scores = sorted(sim_scores, key=lambda x:x[1], reverse=True)
	
	# 가장 코사인 유사도가 높은 10개의 영화와 인덱스를 가져온다. 가장 높은 것은 자기 자신이므로 제외한다.
	sim_scores = sim_scores[1:11]
	movie_indices = [idx[0] for idx in sim_scores]
	return data["title"].iloc[movie_indices]

result = get_recommendations('Tom and Huck')
print(result)