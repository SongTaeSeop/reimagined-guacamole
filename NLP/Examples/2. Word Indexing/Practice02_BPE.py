import re, collections

#%% ##################################################################################
# 현재 dictionary 내 서브워드 pair들의 빈도수 카운트하는 함수

def get_freq(dictionary):
    pairs = collections.defaultdict(int)

    for word, freq in dictionary.items():
        symbols = word.split()

        for i in range(len(symbols)-1):
            pairs[symbols[i], symbols[i+1]] += freq

    print("Frequency of the current pairs", dict(pairs))

    return pairs

#%% ##################################################################################
# target pair를 기준으로 input_dict 내 단어들을 통합하여 output_dict를 반환하는 함수

def merge_dictionary(target_pair, input_dict):
    output_dict = {}

    bigram = re.escape(' '.join(target_pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')

    for word in input_dict:
        w_out = p.sub(''.join(target_pair), word)
        output_dict[w_out] = input_dict[word]

    return output_dict

if __name__ == '__main__':
    num_iter = 10
    dictionary = {'l o w': 5, 
                  'l o w e r': 2,
                  'n e w e s t': 6,
                  'w i d e s t': 3
                }
    vocab = ["l", "o", "w", "e", "r", "n", "s", "t", "w", "i"]

    for i in range(num_iter):
        print(f"{i+1}번째 BPE")
        pairs = get_freq(dictionary)
        best = max(pairs, key=pairs.get)
        dictionary = merge_dictionary(best, dictionary)
        vocab.append("".join(best))
        
        
        print(f"new merge: {best}")
        print(f"dictionary: {dictionary}")
        print(f"vocabulary: {vocab}")
# %%
