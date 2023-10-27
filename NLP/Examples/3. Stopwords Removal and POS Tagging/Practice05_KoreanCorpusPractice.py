#%% ##################################################################################
# 뉴스 기사 크롤링을 위한 패키지 사용
# !pip install newspaper3k

#%%
import newspaper
newspaper.languages()
#%%
from newspaper import Article
URL = "https://www.wikitree.co.kr/articles/813981"
article = Article(URL, language='ko')
article.download()
article.parse()

print('title:', article.title)
print('context:', article.text)

#%% ##################################################################################
# 의도적인 불용어 추가
additional_info = [
  "영상취재 홍길동(hongkildong@chosun.com) 제주방송 임꺽정 홍길동(hongkildong@chosun.com)",
  "<h1>중요한 것은       꺾이지     않는 마음</h1> <h3>카타르 월드컵</h3> <b>유행어</b>로 이루어진    문서입니다."
  "<br>이 줄은 실제 뉴스에 포함되지 않은 임시 데이터임을 알립니다…<br>",
  "Copyright ⓒ JIBS. All rights reserved. 무단 전재 및 재배포 금지.",
  "<이 기사는 언론사 에서 문화 섹션 으로 분류 했습 니다.>",
  "<br>이 줄은 실제 뉴스에 포함되지 않은 임시 데이터임을 알립니다…<br>",
  "#주가 #부동산 #폭락 #환률 #급상승 #중꺾마"
]

context = article.text.split('\n')
context += additional_info

for i, text in enumerate(context):
  print(i, text)
#%% ##################################################################################
# Cleaning - stopwords removal
# 불용어 사전 정의
print("-" * 30+"불용어 사전 정의"+"-" * 30)
stopwords = ['이하', '바로', '☞', '※', '…']

# 불용어 제거
def delete_stopwords(context):
    preprocessed_text = []
    for text in context:
        text = [w for w in text.split(' ') if w not in stopwords]
        preprocessed_text.append(' '.join(text))
    return preprocessed_text

processed_context = delete_stopwords(context)

for i, text in enumerate(processed_context):
    print(i, text)
#%% ##################################################################################
# HTML 태그 제거
print("-" * 30+"HTML 태그 제거"+"-" * 30)
import re
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

processed_context = delete_html_tag(processed_context)

for i, text in enumerate(processed_context):
    print(i, text)
    
#%% ##################################################################################
# 이메일 등 불용어 제거
print("-" * 30+"이메일 등 불용어 제거"+"-" * 30)
def delete_email(context):
  preprocessed_text = []
  for text in context:
    text = re.sub(r"[a-zA-Z0-9+-_.]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "", text).strip()
    if text:
      preprocessed_text.append(text)
  return preprocessed_text

processed_context = delete_email(processed_context)
for i, text in enumerate(processed_context):
    print(i, text)
#%% ##################################################################################
# 해시 태그 제거
print("-" * 30+"해시태그 제거"+"-" * 30)
def delete_hashtag(context):
  preprocessed_text = []
  for text in context:
    text = re.sub(r"#\S+", "", text).strip()
    if text:
      preprocessed_text.append(text)
  return preprocessed_text

processed_context = delete_hashtag(processed_context)
for i, text in enumerate(processed_context):
    print(i, text)
#%% ##################################################################################
# 기타 불용어 제거
print("-" * 30+"기타 불용어 제거"+"-" * 30)
def delete_copyright(context):
  re_patterns = [
    r"\<저작권자(\(c\)|ⓒ|©|\(Copyright\)|(\(c\))|(\(C\))).+?\>",
    r"저작권자\(c\)|ⓒ|©|(Copyright)|(\(c\))|(\(C\))"
  ]
  preprocessed_text = []
  for text in context:
    for re_pattern in re_patterns:
      text = re.sub(re_pattern, "", text).strip()
    if text:
      preprocessed_text.append(text)    
  return preprocessed_text

processed_context = delete_copyright(processed_context)
for i, text in enumerate(processed_context):
    print(i, text)
#%% ##################################################################################
# 형태소 분석
print("-" * 30+"형태소 분석"+"-" * 30)
from konlpy.tag import Okt

okt = Okt()
print("형태소 분석 결과 :")
for sentence in processed_context:
    morphs = okt.morphs(sentence)
    print(morphs)
    
print("품사 태깅 결과 :")
for sentence in processed_context:
    pos = okt.pos(sentence)
    print(pos)