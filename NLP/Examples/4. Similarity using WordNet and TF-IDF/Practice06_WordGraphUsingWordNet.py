#%% ##################################################################################
import nltk
from nltk.corpus import wordnet as wn
# %%
nltk.download('wordnet')
nltk.download('omw-1.4')
#%%
wn.synsets('dog')

#%%
wn.synsets('desk')
# %% ##################################################################################
# 단어의 관계 그래프 확인
man = wn.synset('man.n.01')

for i, path in enumerate(man.hypernym_paths()):
    print(f"[{i}-th path]")
    
    for relation in path:
        print(relation)
        
    print("\n")
# %% ##################################################################################
# 단어간 유사도 계산
man = wn.synset('man.n.01')
boy = wn.synset('boy.n.01')
girl = wn.synset('girl.n.01')
woman = wn.synset('woman.n.01')

print(f"man vs man similarity: {man.path_similarity(man)}")
print(f"man vs boy similarity: {man.path_similarity(boy)}")
print(f"man vs girl similarity: {man.path_similarity(girl)}")
print(f"man vs woman similarity: {man.path_similarity(woman)}")
print(f"boy vs girl similarity: {boy.path_similarity(girl)}")
print(f"boy vs woman similarity: {boy.path_similarity(woman)}")
print(f"girl vs woman similarity: {girl.path_similarity(woman)}")
# %%
