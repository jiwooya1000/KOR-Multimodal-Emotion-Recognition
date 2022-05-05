
##################################################################################
# 바이오 데이터와 Emotion간의 상관성 확인
#
# IBI  : 기록 주기가 불규칙하여 시계열 데이터로 사용 불가
# TEMP : 시계열 데이터로 사용 가능 (누락 데이터 존재)
# EDA  : 시계열 데이터로 사용 가능 (누락 데이터 존재)
#
# 시계열 데이터의 성질을 가지며 길이가 다양한 TEMP와 EDA 데이터에 대하여
# Dynamic Time Warping(DTW)를 활용한 시계열 군집화를 진행한 후,
# 
# 1) 최적의 군집 개수에 대하여 군집화가 유의미하게 이루어지는가?
# 2) Emotion의 가짓수와 동일한 개수로 군집화를 진행할 경우 유의미하게 이루어지는가?
#
# 위 2가지에 대한 평가 진행
#
# 결과
#  - 최적의 군집 개수(n=5)에 대해 군집화가 유의미하지 않다.
#  - Emotion의 가짓수(n=5)에 대해 군집화가 유의미하지 않다.
#  - 따라서, 3가지 생체신호 데이터 모두 Emotion 예측에서 배제한다.
##################################################################################

library(reticulate)

# pickle 파일을 불러오기위해 설치
#py_install('pandas')
#py_install('pickle')
source_python("read_pickle.py")
pickle_data <- read_pickle_file("bio1.pickle")

library(ppls)
pickle_data <- pickle_data[!(pickle_data$TEMP == 'numeric(0)'),]

# pickle data 중 사용하는 감정만 뽑아냄
emotion_data = pickle_data[pickle_data$Emotion == "[0. 0. 0. 1. 0. 0. 0.]" |
                     pickle_data$Emotion == "[1. 0. 0. 0. 0. 0. 0.]"|
                     pickle_data$Emotion == "[0. 1. 0. 0. 0. 0. 0.]"|
                     pickle_data$Emotion == "[0. 0. 1. 0. 0. 0. 0.]"|
                     pickle_data$Emotion == "[0. 0. 0. 0. 0. 0. 1.]",]

# dtwclust에 넣기위해 vector화
temp_list=lapply(emotion_data$TEMP,as.vector)

# max min scale함수를 새로 지정 - 데이터의 변화가 없는 경우는 따로 지정
normalize = function(x){
  if (min(x)== max(x)){
    result = (x-min(x)) / 1
    return(result)
  }else{
  result = (x - min(x)) / (max(x) - min(x))
  return(result)
  }
}

# min max scale적용
normalized_temp= lapply(temp_list,normalize)


library(dtwclust)
library(cluster)

# 클러스터가 5개일 때 지정
cluster=tsclust(normalized_temp, k=5L, distance='dtw_basic')
distance<-dist(normalized_temp, method='dtw_basic')


cl = slot(cluster, "cluster")

# 클러스터가 5개일 때 군집화 plot
plot(cluster, type="sc")

# 각 클러스터 별 centroid 출력
plot(cluster@centroids[[1]])

plot(cluster@centroids[[2]])

plot(cluster@centroids[[3]])

plot(cluster@centroids[[4]])

plot(cluster@centroids[[5]])


install.packages("factoextra")
library(factoextra)

# 최적의 클러스터 개수를 찾기 위해 2~10까지 범위 설정
cluster=tsclust(normalized_temp, k=2L:10L, distance='dtw_basic')
distance<-dist(normalized_temp, method='dtw_basic')

# 실루엣 계수를 확인        
sil = silhouette(x=cluster@cluster, dist=distance)
fviz_silhouette(sil)

# 최적 클러스터를 찾기 위한 파라미터 확인
eval_clust<-sapply(cluster,cvi)
par(mfrow = c(1,2))

# DB index 확인 - 극솟값이 최적의 클러스터 개수
plot(eval_clust[4,],type="l", main="DB index", xlab="The number of clusters",
     ylab="To Be Minimum",col='red',cex.lab=0.8,cex.main=1,axes=F)
axis(1,xlim=c(1.5,5.0),cex.axis=0.8)+axis(2,ylim=c(0,10), cex.axis=0.8)

# Sil index 확인 - 극댓값이 최적의 클러스터 개수
plot(eval_clust[1,],type="l", main="Sil index", xlab="The number of clusters",
     ylab="To Be Maximum",col='red',cex.lab=0.8,cex.main=1,axes=F)
axis(1,xlim=c(0.1,0.4),cex.axis=0.8)+axis(2,ylim=c(0,10),cex.axis=0.8)
