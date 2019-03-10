rm(list=ls(all=TRUE))
setwd('C:/Users/LWS/Desktop/nasa')

library(Rtsne)
library(ggplot2)
library(ggpubr)
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
library(spectral)
library(wmtsa)
library(plotly)
library(ggExtra)
library(tsne)



##### data read #####
name = list.files('C:/Users/LWS/Desktop/nasa/2nd_test/')  # second data set 
file = lapply(paste0('C:/Users/LWS/Desktop/nasa/2nd_test/', name) , fread)  # file name 

sensor.index = 1  # choose sensor 1~4
time.var = sapply(file, function(x) mean(abs(x[[sensor.index]]  )))  # RMS


## RMS plot ##
plot(time.var, type='l', lwd=2, col=rgb(0.2,0.6,0),  xlab=' ', ylab='Root Mean Sqaure Value', xaxt="n", yaxt="n" 
     , panel.first=grid(), main='Sensor data (Bearing1): 2004/02/12 to 2004/02/19', cex.lab=1.5, cex.main=1.5)
ytick<-seq(0, 0.4, by=0.1);axis(side=2, at=ytick, labels = FALSE)
text(par("usr")[1], ytick,labels = ytick, pos=2, xpd = TRUE)
xtick<-seq(1, 1000, by=100);axis(side=1, at=xtick, labels = FALSE)
xlabel = substr(name[xtick],9,20)
xdate = substr(xlabel,1,2);xhour = substr(xlabel,4,5);xmin = substr(xlabel,7,8);xsec = substr(xlabel,10,11)
xlabel = paste0('02/', xdate," ", xhour,':',xmin,':',xsec)
text(xtick, par("usr")[3]-0.03, labels = xlabel, srt=45  , pos=1, xpd = TRUE)

y.true = c( rep(0,533), rep(1,984-533)) # temp y




##### add noise #####
# for (i in 1:length(file)){
#   noise = sapply( file[[i]][[sensor.index]], function(x) runif(1, abs(x)*-0.5 ,  abs(x)*0.5))    # raw +- 2.5%
#   file[[i]][[sensor.index]] = file[[i]][[sensor.index]] + noise
#   print(i)
# }




##### sampling #####
##pacf test ##
# table(sapply(file, function(x) length(x$V1)))
# file.pacf = t(sapply(file, function(x) pacf(x$V1, 10000, plot=FALSE)$acf))
# alpha = 0.1 # %
# interval = qnorm( 1-(alpha/200),0,1)/sqrt(20480)
# min.lag = apply(file.pacf, 1, function(x) max(which(abs(x)>interval)))

select = c(1:2000) # sample index
fft1 = t(sapply(file, function(x)  Mod(fft(scale(x[[ sensor.index ]][select] )))))  # fft amplitude
dim(fft1)

# fft1 = t(sapply(file, function(x)  spectrum(x[[ 1 ]][select])$spec )) 
# dim(fft1)



##### wavelet packet transform test #####
# level = floor(log2(length(select))) 
# fft1 = t(sapply(file, function(x)  unlist(wavDWPT(x$V1[select], n.levels=level)$data ) ))
# fft1 = t(sapply(file, function(x)  summary(wavDWPT(x$V1[select], n.levels=level) )$smat[,10] ))
# 
# fft1 = c()
# for (j in 1:984){
#   sft = c()
#   for (i in 1:100){
#     select = c(1:4000)+160*i
#     sft = rbind(sft, as.vector(Mod(fft(scale(  file[[ j ]]$V1[ select ] ))) ))
#   }
#   sft.avg = apply(sft, 2, function(x) mean(x))
#   fft1 = rbind(fft1, sft.avg)
#   print(j)
# }




##### smoothing #####
sm1 = t(apply(fft1, 1, function(x) smooth.spline(x,spar = 0.2)$y))
dim(sm1)

## normalize
for (i in 1:nrow(sm1)){
  dat = sm1[i,]
  sm1[i,] = (dat - min(dat))/((max(dat)-min(dat)))
}



##### train/test set #####
split = 533
train.set = c(1:split)
test.set = c((split+1):length(file))
# train.set = c(534:(534+168))
# test.set = c(1:533,(534+169):length(file))
# train.set = sort(sample(c(1:length(file)), split ))
# test.set = c(1:length(file))[-train.set]

train.sm = sm1[train.set, ]
test.sm = sm1[test.set, ]
dim(train.sm); dim(test.sm)

train.y = y.true[train.set]
test.y = y.true[test.set]

# train.x <- lle( X=sm1[train.set, ], m=30, k=10)
# test.x <- lle( X=sm1[test.set, ], m=30, k=10)



##### modeling #####
# pca1 = kpca(train.sm, kernel='rbfdot', sigma=0.0030, features=10 )@pcv  # kernel pca
# ts.plot(pca1[,1])

pca1 = prcomp( train.sm )  # PCA
min(which((cumsum(pca1$sdev)/sum(pca1$sdev) > 0.8 )))  
cut.pc = min(which((cumsum(pca1$sdev)/sum(pca1$sdev) > 0.8 )))  # PC number

train.score = pca1$x   # train score
train.load = pca1$rotation # train loading
# train.avg = t(matrix( pca1$center , length(pca1$center), nrow(pca1$x)))  # center
# check = (train.score %*% t(train.load)) - (train.sm-train.avg); ts.plot(apply(check, 2, function(x) sum(x)))
# dim(train.score); dim(t(train.load)); dim(train.avg)

test.avg = t(matrix( pca1$center , length(pca1$center), nrow( test.sm )))  # centering
test.score = (( test.sm - test.avg)) %*% ginv(t(train.load))  # test score
# check = (test.score %*% t(train.load)) - (test.sm-test.avg); ts.plot(apply(check, 2, function(x) sum(x)))
# dim(test.score); dim(t(train.load)); dim(test.avg)

train.x = train.score[,c(1:cut.pc)]  # train score
test.x = test.score[,c(1:cut.pc)]    # test score
dim(train.x); dim(test.x)

# train.centroid = apply(train.x[which(train.y==0),], 2, function(x) median(x))
train.centroid = apply(train.x, 2, function(x) median(x))  # centroid 
# weight = apply(train.x[which(train.y==0),], 2, function(x) sd(x))

train.dist = c()
for (i in 1:nrow(train.x)){
  train.dist = c(train.dist, sum(abs(train.centroid-train.x[i,]) ))  # train distance
}

test.dist = c()
for (i in 1:nrow(test.x)){
  test.dist = c(test.dist, sum(abs(train.centroid-test.x[i,]) ))  # test distance 
}

ts.plot(c(train.dist, test.dist));lines(time.var*50+5,col='blue')




##### health index ##### 
health.index =  c(train.dist, test.dist)
health.index = (health.index) / mean(train.dist)  
health.index = 1 - 1/health.index
health.index[which(health.index<0)] = 0  

health.index = smooth.spline(health.index, spar=0.2)$y

# 1 - 1/(mean(train.dist)+1*sd(train.dist))
# 1 - 1/(mean(train.dist)+3*sd(train.dist))
# 1 - 1/(mean(train.dist)+6*sd(train.dist))


## health index plot ##
plot(health.index*100, type='l', lwd=2,
     col=rgb(0.1,0.1,0.6), ylim=c(0,100),
     xlab=' ', ylab='Health index(%)', xaxt="n", yaxt="n" 
     , panel.first=grid(), main='Health index: 2004/02/12 to 2004/02/19', cex.lab=1.5, cex.main=1.5)
ytick<-seq(0, 100, by=20);axis(side=2, at=ytick, labels = FALSE)
text(par("usr")[1], ytick,labels = ytick, pos=2, xpd = TRUE)
xtick<-seq(1, 1000, by=100);axis(side=1, at=xtick, labels = FALSE)
xlabel = substr(name[xtick],9,20)
xdate = substr(xlabel,1,2);xhour = substr(xlabel,4,5);xmin = substr(xlabel,7,8);xsec = substr(xlabel,10,11)
xlabel = paste0('02/', xdate," ", xhour,':',xmin,':',xsec)
text(xtick, par("usr")[3]-8, labels = xlabel, srt=45  , pos=1, xpd = TRUE)


## 2 axis plot ##
par(new=TRUE)
plot( c(train.dist, test.dist)  ,type="l",col=rgb(1,0,0,0.3),xaxt="n",yaxt="n",xlab="",ylab="")
axis(4)
mtext("Feature Space",side=4,line=3, col='red', cex=1.5)
legend("topleft",col=c("red",rgb(0.1,0.1,0.6)),lty=1,legend=c("Feature Space","Health Index"))


## density plot ##
ts.plot(train.dist); abline(v=min(which(train.y==1)),col='red')
ts.plot(test.dist); abline(v=min(which(test.y==1)),col='red')

hist((train.dist), breaks=seq(0, max(c(train.dist,test.dist)),by=max(c(train.dist,test.dist))/100), col=rgb(1,0,0,0.3), xlim=c(min(train.dist,test.dist),max(train.dist,test.dist)), main='Overlapping Histogram')
hist((test.dist),  breaks=seq(0, max(c(train.dist,test.dist)),by=max(c(train.dist,test.dist))/100), col=rgb(0,0,1,0.3), add=T)


## 3d plot ##
data.3d = as.data.frame( rbind(train.x[,c(1:3)], test.x[,c(1:3)])   )
colnames(data.3d) = c('Axes1','Axes2','Axes3')
data.3d$Condition = as.factor( c(rep('Normal', nrow(train.x)), rep('Test', nrow(test.x))) )
data.3d$case2 = seq(1, nrow(data.3d))/nrow(data.3d)
colfunc <- colorRampPalette(c('gold'   , 'red'  ))
# data.3d$case3 = c(rep("#00AFBB", 533), rep("slateblue", 168), colfunc(283)  )
data.3d$case3 = c(rep("red", 533), rep("green", 168), rep("blue", 283)  )

p <- plot_ly(data.3d, x = ~Axes1, y = ~Axes2, z = ~Axes3, 
             marker = list(color = ~case3, size=6
                           # , colorscale = c('#FFE1A1', '#683531')
                           ,opacity = 0.2, showscale = F)) %>% 
  add_markers() %>%
  layout(scene = list(xaxis = list(title = 'Axes1'),
                      yaxis = list(title = 'Axes2'),
                      zaxis = list(title = 'Axes3')))  
p

# library(smoothAPC)
# tt = time.var/max(time.var)
# plot3d( cbind(  sm1[ , c(700:3000)] ))


## Grouped Scatter plot with marginal density plots ##
ggscatterhist(
  data.3d, x = "Axes1", y = "Axes2",
  color = "Condition", size = 4, alpha = 0.15,
  palette = c("#00AFBB", "#E7B800"),
  margin.params = list(fill = "Condition", color = "black", size = 0.1 )
)



## 2d density ##
ggplot(data.3d , aes(Axes1, Axes2)) + 
  geom_point(data = data.3d , size=4, color= c(rep("#00AFBB",533) , rep("#E7B800",168 ), rep(rgb(0.7,0.7,0.7), 984-533-168))
             , alpha=0.15 ) +
  # xlim(-4.4, 0.3) +
  # ylim(-1.1, 0.35) +
  # 533, 451 = 984
  # 533, 168, 250 
  stat_density2d(aes(alpha=..level.., fill=..level..), size=30, 
                 bins= 300, geom="polygon") + 
  scale_fill_gradient(low = rgb(0.1,0.9,0.2), high = rgb(0.8,0.1,0.5)  ) +
  scale_alpha(range = c(0.00, 0.04), guide = FALSE) +
  geom_density2d(colour= rgb(0.2,0.2,0.2,0.3), bins=5) +
  theme_bw() + 
  theme(axis.line=element_blank(),
        axis.text.x=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks=element_blank(),
        axis.title.x=element_blank(),
        axis.title.y=element_blank(),
        # legend.position="none",
        panel.background=element_blank(),
        # panel.border=element_blank(),
        panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),
        plot.background=element_blank()) + theme(legend.position = c(0.9, 0.3))



## t sne ##
trn <- data.matrix( rbind(train.x, test.x ))
# cols <- c(rep('red',533), rep('blue',451))
# ecb = function(x, y){ plot(x, t='n'); text(x, labels=c(1:nrow(trn)) , col=cols); }
# tsne_res = tsne(trn , k=2, epoch_callback = ecb , perplexity = 2, max_iter = 200 )

tsne_model_1 = Rtsne(as.matrix(trn), check_duplicates=FALSE, pca=TRUE, perplexity=20, theta=0.3, dims=2)
d_tsne_1 = as.data.frame(tsne_model_1$Y) 
# plot(d_tsne_1$V1, d_tsne_1$V2)

ggplot(d_tsne_1 , aes(V1, V2, label =   round( c(1:984)/100,1)  )) + 
  geom_point(data = d_tsne_1, size=4, 
             color = c(rep("#00AFBB",533) , rep("#E7B800",168 ), rep('black', 984-533-168)), alpha=0.15) +
  geom_text(aes(  color = c(rep("#00AFBB",533) , rep("#E7B800",168 ), rep('black', 984-533-168))), size= 4.5 ) +
  # geom_label_repel(aes(label = c(1:984),
  #                      fill = c(rep("#00AFBB",533) , rep("#E7B800",168 ), rep('black', 984-533-168)) )
  #                     , color = 'white',size = 1) +
  # xlim(-100,100)+
  # ylim(-100,100) + 
  theme(legend.position = "none")



## cor plot ##
library(ggcorrplot)
corr.data <- cor(cbind(a1,a2,a3,a4,a5,a6,a7))
colnames(corr.data) = c('10%', '20%', '30%', '40%', '60%', '80%', '100%')
rownames(corr.data) = c('10%', '20%', '30%', '40%', '60%', '80%', '100%')
ggcorrplot(corr.data, hc.order = FALSE, type = "lower", lab = TRUE,
           outline.col = "white")
corr.data2 = rbind(  c(1, 0.99, 0.92, 0.91) ,
                     c(0.99, 1, 0.91, 0.90) ,
                     c(0.95, 0.94, 1, 0.91) ,
                     c(0.91, 0.92, 0.91, 1) )
colnames(corr.data2) = c('10%', '30%', '50%', '100%')
rownames(corr.data2) = c('10%', '30%', '50%', '100%')
ggcorrplot(corr.data2, hc.order = FALSE, type = "lower", lab = TRUE,
           outline.col = "white")




##### health index AUC test #####
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


