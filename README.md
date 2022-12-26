# KOR-Multimodal-Emotion-Recognition
2022 휴먼이해 인공지능 논문경진대회 우수상 수상 (논문)[https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11113938&language=ko_KR&hasTopBanner=true]의 코드를 정리한 Repository입니다. 


_**각성도 및 긍/부정도의 싱글모달 사전학습 예측 모델 기반 멀티모달 감정인식 모델**_

_**Multimodal Emotional Recognition Model based on Singlemodal Pretrained Prediction Model of Valence and Arousal**_


음성과 텍스트를 기반으로 감정 분류를 예측하기 위한 멀티모달 딥러닝 모델을 구현하였으며, 사람의 발화 음성을 바탕으로 각성도(Arousal)를 예측하는 모델과 발화 텍스트를 바탕으로 긍/부정도(Valence)를 예측하는 모델을 개별적으로 사전학습하여 멀티모달 감정 분류 모델의 성능을 향상시켰습니다.

**보다 구체적인 분석 과정을 확인하려면, *Multimodal Emotional Recognition Model KOR.pdf*를 참고해주세요.**

----
### 코드 진행 순서 (Updated (5/8))
  - 1-1.데이터_취합.ipynb를 제외한 모든 jupyter notebook 파일은 Google Colaboratory를 기반으로 작성되었습니다.
  - 따라서, KEMDy20 데이터셋이 존재하는 로컬 디렉토리에서 1-1. 데이터_취합.ipynb를 실행하여 Data_Original.pickle을 생성한 후 나머지 Google Colaboratory 기반으로 작성된 .ipynb가 존재하는 Google Drive 디렉토리에 업로드해야 합니다.
  - 아래 순서에 따라 실행했을 경우 최종 반환되는 Audio-Arousal 모델 / Text-Valence 모델 / Multimodal Emotion Classifier 모델의 .pt 파일은 아래의 링크를 통해 확인 가능합니다.
  - https://drive.google.com/drive/folders/1FPg_OvxY1ADOSWq9R_ln0LzP4CMl2knk?usp=sharing
#### 1. Data Preprocessing
      - 1-1. 데이터_취합.ipynb 실행 (Data_Original.pickle 파일 생성)
      - 1-2. 데이터_증강.ipynb 실행 (train_aft_aug_kobert.pickle, valid_tokenized.pickle, test_tokenized.pickle 생성)
      - 1-3. 시계열 군집화.R 실행 (bio1.pickle 사용)
#### 2. Audio-Arousal Model
      - 2-1. Audio_Arousal Model Train.ipynb 실행(audio_arouosal.pt 생성)
#### 3. Text-Valence Model
      - 3-1. Text_Valence Model Train.ipynb 실행(text_valence.pt 생성)
#### 4. Multimodal Emotion Classifier
      - 멀티모달_학습.ipynb 실행(Multi_Modal_Classifier_ye.pt 생성)
----

## 1. Data Preprocessing


###  (1) 음성 데이터
    - Mel Spectrogram과 Mel Spectrogram의 1차 차분값을 stack하여 사용
    - Mel Spectrogram의 길이를 일정하게 통일시키기 위해 Zero Padding 적용
    - 데이터 불균형 보완 및 완화를 위해 Random Frequency Masking 적용 
 
###  (2) 텍스트 데이터
    - KoBERT의 tokenizer를 활용하여 토큰화
    - 단어 간 랜덤 위치 변환(Random Swap), 텍스트 중 단어 임의 삭제(Random Delete) 적용
    
###  (3) 생체신호 데이터
    - IBI : 기록 주기가 불규칙하여 사용하지 않음
    - TEMP: 기록 주기가 규칙적이나 시계열 군집화 결과 감정 분류와 유의미한 관계 없음
    - EDA : 기록 주기가 규칙적이나 누락된 Session이 존재하여 사용하지 않음

###  (4) 데이터 불균형 보완
    - 음성: Random Frequency Masking
    - 텍스트: Random Swap, Random Deletion


## 2. Audio-Arousal Model

###  모델 구조
    - Kernel Size를 (1, 4)로 설정하여 Frequency 방향으로는 인접한 Frequency 간의 관계적 특징을 추출
    - Time 방향으로는 시간 정보를 독립적으로 유지하도록 특징 추출
    - Multihead Attention + GRU
    - Multihead Attention 학습 레이어를 반복하여 쌓아 Self Attention 계산
    - 시간 정보를 독립적으로 유지하였기 때문에 GRU를 통해 시간적 정보를 고려한 Arousal 예측 진행


## 3. Text-Arousal Model
###  모델 구조
    - KoBERT Embedding
      - 사전학습된 KoBERT 모델을 기반으로 각 단어를 768차원의 벡터로 임베딩
    - Multihead Attention + GRU
      - Multihead Attention 학습 레이어를 반복하여 쌓아 Self Attention 계산
      - BERT의 특성상 Positional Embedding을 통해 순서 정보가 내재되어 있으므로 GRU를 통해 시간적 정보를 고려한 Valence 예측 진행


## 4. Multimodal Emotion Classifier
###  모델 구조
    - Audio
      - Arousal을 예측하도록 학습된 AudioRegressor 모델의 예측 레이어 이전 Self Attention Value를 Audio Embedded Vector로 사용
    - Text
      - Valence를 예측하도록 학습된 TextRegressor 모델의 예측 레이어 이전 Self Attention Value를 Text Embedded Vector로 사용
    - Classifier
      - Audio Embedded Vector : torch.Size([batch_size, 512, 768])
      - Text Embedded Vector : torch.Size([batch_size, 64, 768])
      - Audio와 Text의 Embedded Vector를 연결하여 torch.Size([batch_size, 576, 768]) tensor 생성
      - Multihead Attention을 통한 Self Attention 학습
      - 최종 반환된 Attention Value을 LSTM에 통과시켜 감정 분류 진행

 
