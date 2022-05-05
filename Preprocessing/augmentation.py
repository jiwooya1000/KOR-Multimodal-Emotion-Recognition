import random
import numpy as np

"""
<<입력 형식>>

text: 토큰화가 완료된 list
frac: 텍스트에 해당 함수를 적용할 비율
wordnet: 동의어들을 모아놓은 wordnet (http://wordnet.kaist.ac.kr/)
mel_spec: 전처리된 [Mel Spectrogram + 1차 차분값]
"""


# 텍스트의 일정 비율을 동의어로 바꿔주는 함수

def SR(input: list, frac: float, wordnet, seed=824):
    random.seed(seed)
    text = input.copy()

    # 동의어로 교체할 단어 수 지정    
    if round(len(text) * frac) > 1:
        replace_num = round(len(text) * frac)
    else:
        replace_num = 1

    index = random.sample(range(len(text)), replace_num)

    for i in index:
        word = text[i]
        for j in range(wordnet.shape[0]):
            syn = wordnet.iloc[j, 3].split(', ')
            if word in syn:
                if len(syn) <= 1:
                    pass
                else:
                    syn.remove(word)
                    replace = random.choice(syn)
                    text[i] = replace
            else:
                pass

    return text


# 텍스트의 일정 비율을 제거해주는 함수

def RD(input: list, frac: float, seed=824):
    random.seed(seed)
    text = input.copy()

    # 제거할 단어 수 지정
    if round(len(text) * frac) > 1:
        del_num = round(len(text) * frac)
    else:
        del_num = 1

    index = random.sample(range(len(text)), del_num)
    for i in sorted(index, reverse=True):
        del text[i]

    return text


# 텍스트의 일정 비율의 단어들의 순서를 바꿔줌

def RS(input: list, frac: float, seed=824):
    random.seed(seed)
    text = input.copy()

    if len(text) == 1:
        pass
    else:
    # 위치를 바꿔줄 단어 수 지정
        if round(len(text) * frac) > 2:
            swap_num = round(len(text) * frac)
        else:
            swap_num = 2

        index = random.sample(range(len(text)), swap_num)
        swap_list = []

        for i in range(len(index) - 1):
            text[index[i]], text[index[i + 1]] = text[index[i + 1]], text[index[i]]

    return text


# 음성 Mel_Spectrogram Frequency Masking 적용해주는 함수

def audio_mask(mel_spec, mask_freq, mask_range, freq_range, seed=824):
    np.random.seed(seed)
    data = mel_spec.copy()
    freq_range.sort()
    start = mask_freq
    pad_start = 512

    for i in range(data.shape[1]):
        if np.all(data[0, i, :] == 0):
            pad_start = i
            break

    if mask_freq + mask_range >= 128:
        end = 128
        mask_range = end - mask_freq
    else:
        end = mask_freq + mask_range

    data[:, :pad_start, start:end] = np.random.uniform(freq_range[0], freq_range[1],
                                                       size=(data.shape[0], pad_start, mask_range))

    return data
