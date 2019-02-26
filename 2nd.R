rm(list=ls(all=TRUE))

setwd('C:/Users/LWS/Desktop/nasa')
library(data.table)
library(TSA)
library(wavelets)
library(xgboost)
library(MLmetrics)
library(caret)
library(overlapping) 
library(BBmisc)
library(MASS)
library(geosphere)
library(nortest)
library(pROC)




##### read #####
name = list.files('C:/Users/LWS/Desktop/nasa/2nd_test/')        # second data set 
file = lapply(paste0('C:/Users/LWS/Desktop/nasa/2nd_test/', name) , fread)
raw.file = file
time.var = sapply(raw.file, function(x) sd(x$V1))             
ts.plot(time.var)                                               # time series data: deviation 

file = raw.file[ c(1:300, 801:900)  ]                           # choosing data set
time.var = sapply(file, function(x) sd(x$V1))             
ts.plot(time.var)

y.true = c( rep(0,689), rep(1,295))                             # y




##### add noise #####
for (i in 1:length(y.true)){
  noise = sapply( file[[i]]$V1, function(x) runif(1, abs(x)*-0.025 ,  abs(x)*0.025))    # raw +- 2.5%
  file[[i]]$V1 = file[[i]]$V1 + noise
  print(i)
}




##### sampling #####
# table(sapply(file, function(x) length(x$V1)))
# file.pacf = t(sapply(file, function(x) pacf(x$V1, 10000, plot=FALSE)$acf))
# alpha = 0.1 # %
# interval = qnorm( 1-(alpha/200),0,1)/sqrt(20480)
# min.lag = apply(file.pacf, 1, function(x) max(which(abs(x)>interval)))


select = c(1:4000)
fft1 = t(sapply(file, function(x) Mod(fft(scale(x[[ 1 ]][select] )))))
dim(fft1)
ts.plot(fft1[1,])
ts.plot(fft1[300,])
ts.plot(fft1[350,])
ts.plot(fft1[380,]-fft1[350,])
ts.plot(fft1[380,]-fft1[1,])




##### smoothing #####
sm1 = c()
for (i in 1:length(file)){
  smoothingSpline = smooth.spline(fft1[i ,], spar=0.2)
  y = smoothingSpline$y
  sm1 = rbind(sm1, y)
  print(i)
}
dim(sm1)
ts.plot(sm1[1, ])
ts.plot(sm1[300, ])
ts.plot(sm1[350, ])
ts.plot(sm1[380,]-sm1[350,])
ts.plot(sm1[380,]-sm1[1,])


# fft2 = apply(sm1, 2, function(x) Mod(fft(x)))
# dim(fft2)
# ts.plot(fft2[1,])
# ts.plot(fft2[750,])
# ts.plot(fft2[950,])
# ts.plot(fft2[950,]-fft2[750,])
# ts.plot(fft2[950,]-fft2[1,])




##### train/test set #####
split = 689
# train.set = c(1:split)
# test.set = c((split+1):length(file))
train.set = sort(sample(c(1:length(file)), split ))
test.set = c(1:length(file))[-train.set]

train.sm = sm1[train.set, ]
test.sm = sm1[test.set, ]
dim(train.sm); dim(test.sm)

train.y = y.true[train.set]
test.y = y.true[test.set]




##### modeling #####
pca1 = prcomp( train.sm )

ts.plot(cumsum(pca1$sdev)/sum(pca1$sdev))
ts.plot((cumsum(pca1$sdev)/sum(pca1$sdev))[1:100]); abline(h=0.9,col='blue')
min(which((cumsum(pca1$sdev)/sum(pca1$sdev) > 0.9 )))  
cut.pc = min(which((cumsum(pca1$sdev)/sum(pca1$sdev) > 0.9 )))

train.score = pca1$x
train.load = pca1$rotation
train.avg = t(matrix( pca1$center , length(pca1$center), nrow(pca1$x)))
# check = (train.score %*% t(train.load)) - (train.sm-train.avg); ts.plot(apply(check, 2, function(x) sum(x)))
dim(train.score); dim(t(train.load)); dim(train.avg)

test.avg = t(matrix( pca1$center , length(pca1$center), nrow( test.sm )))
test.score = (( test.sm - test.avg)) %*% ginv(t(train.load))
# check = (test.score %*% t(train.load)) - (test.sm-test.avg); ts.plot(apply(check, 2, function(x) sum(x)))
dim(test.score); dim(t(train.load)); dim(test.avg)

# dim(test.avg); dim(test.score)
# start_time <- Sys.time()
# end_time <- Sys.time()
# end_time - start_time

train.x = train.score[,c(1:cut.pc)]
test.x = test.score[,c(1:cut.pc)]
dim(train.x); dim(test.x)

train.centroid = apply(train.x, 2, function(x) median(x))

train.dist = c()
for (i in 1:nrow(train.x)){
  train.dist = c(train.dist, sum(abs(train.centroid-train.x[i,])))
}

test.dist = c()
for (i in 1:nrow(test.x)){
  test.dist = c(test.dist, sum(abs(train.centroid-test.x[i,])))
}

ts.plot(train.dist); abline(v=min(which(train.y==1)),col='red')
ts.plot(test.dist); abline(v=min(which(test.y==1)),col='red')

hist(train.dist, breaks=seq(0,max(train.dist)+100,by=80), ylim=c(0,60), col=rgb(1,0,0,0.3), xlim=c(min(train.dist),max(test.dist)), main='Overlapping Histogram')
hist(test.dist, breaks=seq(0,max(test.dist)+100,by=80), col=rgb(0,0,1,0.3), add=T)




##### health index #####
my_roc <- roc(train.y, train.dist)
cut.off = coords(my_roc, "best", ret = "threshold")
train.pred2 = ifelse(train.dist>cut.off,1,0)
test.pred2 = ifelse(test.dist>cut.off,1,0)



pre.re = function(x,y){
  tp = length(which(x==1 & y==1))
  fp = length(which(x==1 & y==0))
  fn = length(which(x==0 & y==1))
  pre = tp/(tp+fp)
  re = tp/(tp+fn)
  out = list('prediction'=pre, 'recall'=re)
  return(out)
}

pre.re(train.pred2, train.y)
pre.re(test.pred2, test.y)



