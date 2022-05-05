# KOR-Multimodal-Emotion-Recognition
2022 휴먼이해 인공지능 논문경진대회 참여 논문 *논문 제목*의 코드를 정리한 Repository입니다. 
음성과 텍스트를 기반으로 감정 분류를 예측하기 위한 멀티모달 딥러닝 모델을 구현하였으며, 사람의 발화 음성을 바탕으로 각성도(Arousal)를 예측하는 모델과 발화 텍스트를 바탕으로 긍/부정도(Valence)를 예측하는 모델을 개별적으로 사전학습하여 멀티모달 감정 분류 모델의 성능을 향상시켰습니다.


## 1. Data Preprocessing
1. 활용 데이터 선정 및 처리
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
3. 
