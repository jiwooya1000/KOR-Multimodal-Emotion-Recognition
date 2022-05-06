# KOR-Multimodal-Emotion-Recognition
2022 휴먼이해 인공지능 논문경진대회 참여 논문 *논문 제목*의 코드를 정리한 Repository입니다. 
음성과 텍스트를 기반으로 감정 분류를 예측하기 위한 멀티모달 딥러닝 모델을 구현하였으며, 사람의 발화 음성을 바탕으로 각성도(Arousal)를 예측하는 모델과 발화 텍스트를 바탕으로 긍/부정도(Valence)를 예측하는 모델을 개별적으로 사전학습하여 멀티모달 감정 분류 모델의 성능을 향상시켰습니다.

----
### 코드 진행 순서
#### 1. Data Preprocessing
#### 2. Audio-Arousal Model
#### 3. Text-Valence Model
#### 4. Multimodal Emotion Classifier
----

## 1. Data Preprocessing
### 활용 데이터 선정 및 처리
  - 음성 데이터
    * Mel Spectrogram과 Mel Spectrogram의 1차 차분값을 stack하여 사용
    * Mel Spectrogram의 길이를 일정하게 통일시키기 위해 Zero Padding 적용
    * 데이터 불균형 보완 및 완화를 위해 Random Frequency Masking 적용
 
![image](https://user-images.githubusercontent.com/20739007/167010497-0df6fd38-8542-4909-a513-5ed72c0d63df.png)
 
 
  - 텍스트 데이터
    * KoBERT의 tokenizer를 활용하여 토큰화
    * 단어 간 랜덤 위치 변환(Random Swap), 텍스트 중 단어 임의 삭제(Random Delete) 적용


  - 생체신호 데이터
    * IBI : 기록 주기가 불규칙하여 사용하지 않음
    * TEMP: 기록 주기가 규칙적이나 시계열 군집화 결과 감정 분류와 유의미한 관계 없음
    * EDA : 기록 주기가 규칙적이나 누락된 Session이 존재하여 사용하지 않음

![image](https://user-images.githubusercontent.com/20739007/167011063-d0904346-90ec-4062-87ea-dd86d2307ccc.png) 

### 데이터 불균형 보완
  - 음성: Random Frequency Masking
  - 텍스트: Random Swap, Random Deletion
  - 데이터가 많은 happy와 neutral을 제외한 나머지 3개의 감정(sad, angry, surprise)에 대해 음성 증강, 텍스트 증강, 음성과 텍스트 증강을 각각 적용하여 250%까지 증강

![image](https://user-images.githubusercontent.com/20739007/167014667-f73efc68-8c53-4534-a26d-f79afc78df64.png)


## 2. Audio-Arousal Model
### 모델 구조
  - AudioExtractor
    * Kernel Size를 (1, 4)로 설정하여 Frequency 방향으로는 인접한 Frequency 간의 관계적 특징을 추출하고, Time 방향으로는 시간 정보를 독립적으로 유지하도록 특징 추출

![audio_arousal](https://user-images.githubusercontent.com/20739007/167012301-37877c91-e955-40f3-ae57-340ac0bccab6.png)

  - Multihead Attention + GRU
    * Multihead Attention 학습 레이어를 반복하여 쌓아 Self Attention 계산
    * 시간 정보를 독립적으로 유지하였기 때문에 GRU를 통해 시간적 정보를 고려한 Arousal 예측 진행


## 3. Text-Arousal Model
### 모델 구조
  - KoBERT Embedding
    * 사전학습된 KoBERT 모델을 기반으로 각 단어를 768차원의 벡터로 임베딩
  - Multihead Attention + GRU
    * Multihead Attention 학습 레이어를 반복하여 쌓아 Self Attention 계산
    * BERT의 특성상 Positional Embedding을 통해 순서 정보가 내재되어 있으므로 GRU를 통해 시간적 정보를 고려한 Valence 예측 진행


## 4. Multimodal Emotion Classifier
### 모델 구조
  - Audio
    * Arousal을 예측하도록 학습된 AudioRegressor 모델의 예측 레이어 이전 Self Attention Value를 Audio Embedded Vector로 사용
  - Text
    * Valence를 예측하도록 학습된 TextRegressor 모델의 예측 레이어 이전 Self Attention Value를 Text Embedded Vector로 사용
  - Classifier
    * Audio Embedded Vector : torch.Size([batch_size, 512, 768])
    * Text Embedded Vector : torch.Size([batch_size, 64, 768])
    * Audio와 Text의 Embedded Vector를 연결하여 torch.Size([batch_size, 576, 768]) tensor 생성
    * Multihead Attention을 통한 Self Attention 학습
    * 최종 반환된 Attention Value을 LSTM에 통과시켜 감정 분류 진행

 
