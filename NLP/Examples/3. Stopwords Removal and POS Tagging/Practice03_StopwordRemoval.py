#%% ##################################################################################
import nltk
nltk.download('punkt')
nltk.download('stopwords')

#%% ##################################################################################
from nltk.corpus import stopwords

english_stopwords = stopwords.words('english')

print(f"The number of total stopwords: {len(english_stopwords)}")
print(*english_stopwords, sep='\n')

#%% 원하는 불용어만 제거

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

eng_sample = "Back translation is a quality assurance technique that can add clearity to and control over your translated content!"
stop_words = set(stopwords.words('english'))

tokens = word_tokenize(eng_sample)

cleaned_tokens = []

for w in tokens:
    if w not in stop_words:
        cleaned_tokens.append(w)

print(tokens, '\n')
print(cleaned_tokens)

#%% ##################################################################################
# 한국어 불용어 처리

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

kor_sample = "한국어 전처리 작업에서는 주로 불용어 단어로 조사, 접속사 등을 사용합니다."
stop_words = ["에서는", "등을", "합니다", ".", ","]

tokens = word_tokenize(kor_sample)

cleaned_tokens = []

for w in tokens:
    if w not in stop_words:
        cleaned_tokens.append(w)

print(tokens, '\n')
print(cleaned_tokens)

# %% ##################################################################################
# 불필요한 태그 및 특수 문자 제거
import re
p = re.compile('^[a-zA-Z0-9+-_.]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$')

emails = ['python@mail.example.com',
          'testtest@naver.com',
          'lululala@n-aver.com',
          '@example.com'
          ]

for email in emails:
    print(p.match(email) != None, end=' ')

#%% ##################################################################################
# HTML 태그 제거
def delete_html_tag(context):
  """
  ex. <h1>뉴스 제목</h1> -> 뉴스 제목
  """
  preprcessed_text = []

  for text in context:
      text = re.sub(r"<[^>]+>\s+(?=<)|<[^>]+>", "", text).strip()
      if text:
          preprcessed_text.append(text)
  return preprcessed_text

text = [
  "영상취재 홍길동(hongkildong@chosun.com) 제주방송 임꺽정 홍길동(hongkildong@chosun.com)",
  "<h1>중요한 것은       꺾이지     않는 마음</h1> <h3>카타르 월드컵</h3> <b>유행어</b>로 이루어진    문서입니다."
  "<br>이 줄은 실제 뉴스에 포함되지 않은 임시 데이터임을 알립니다…<br>",
  "Copyright ⓒ JIBS. All rights reserved. 무단 전재 및 재배포 금지.",
  "<이 기사는 언론사 에서 문화 섹션 으로 분류 했습 니다.>",
  "<br>이 줄은 실제 뉴스에 포함되지 않은 임시 데이터임을 알립니다…<br>",
  "#주가 #부동산 #폭락 #환률 #급상승 #중꺾마"
]
cleaned_text = delete_html_tag(text)

for i, line in enumerate(cleaned_text):
    print(i, line)
# %% ##################################################################################
# WordNet Lemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.stem import WordNetLemmatizer

lemmatier = WordNetLemmatizer()

print(lemmatier.lemmatize('dies', 'v'))
print(lemmatier.lemmatize('watched', 'v'))

# %%
