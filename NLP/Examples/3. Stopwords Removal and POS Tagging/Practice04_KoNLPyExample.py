#%% ##################################################################################
!pip install konlpy

#%% ##################################################################################
# konlpy 관련 패키지 import
from konlpy.tag import Okt, Kkma, Komoran

okt = Okt()
kkma = Kkma()
komoran = Komoran()

#%% ##################################################################################
# konlpy 중 Kkma는 문장 분리가 가능 (다른 라이브러리는 되지 않음)
print ("kkma 문장 분리 : ", kkma.sentences('네 안녕하세요 반갑습니다.'))

#%% ##################################################################################
# konlpy 의 라이브러리 형태소 분석 비교
print("okt 형태소 분석 :", okt.morphs(u"집에 가면 감자 좀 쪄줄래?"))
print("kkma 형태소 분석 : ", kkma.morphs(u"집에 가면 감자 좀 쪄줄래?"))
print("komoran 형태소 분석 : ", komoran.morphs(u"집에 가면 감자 좀 쪄줄래?"))

#%% ##################################################################################
# konlpy 의 라이브러리 품사태깅 비교
print("okt 품사태깅 :", okt.pos(u"집에 가면 감자 좀 쪄줄래?"))
print("kkma 품사태깅 : ", kkma.pos(u"집에 가면 감자 좀 쪄줄래?"))
print("komoran 품사태깅 : ", komoran.pos(u"집에 가면 감자 좀 쪄줄래?"))
# %% ##################################################################################
# 띄어쓰기가 안된 문장
sample = "아빠가방에들어가신다"
print("okt :", okt.pos(sample))
print("kkma : ", kkma.pos(sample))
print("komoran : ", komoran.pos(sample))
# %% ##################################################################################
# 오타가 있는 문장
sample = "오늘 뭐햇니? 나는 우너하던 일을 했어!"
print("okt :", okt.pos(sample))
print("kkma : ", kkma.pos(sample))
print("komoran : ", komoran.pos(sample))

# %%
