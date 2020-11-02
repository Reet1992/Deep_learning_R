sum.of.squares <- function(x,y)
{
  x^2 + y^2
}




logit <- function(p)
{
 log(p) - log(1 - p)
  
}




library(mlbench)

data("BreastCancer")

#Clean off rows with missing data
BreastCancer = BreastCancer[which(complete.cases(BreastCancer)==TRUE),]

head(BreastCancer)


names(BreastCancer)


#### Specify input and the output #### 


y = as.matrix(BreastCancer[,11])
y[which(y=="benign")] = 0
y[which(y=="malignant")] = 1
y = as.numeric(y)
x = as.numeric(as.matrix(BreastCancer[,2:10]))
x = matrix(as.numeric(x),ncol=9)


### Model Creation 

library(deepnet)

nn <- nn.train(x, y, hidden = c(5))
yy = nn.predict(nn, x)

### creation of confusion matrix and determination the accuracy


yhat = matrix(0,length(yy),1)
yhat[which(yy > mean(yy))] = 1
yhat[which(yy <= mean(yy))] = 0
cm = table(y,yhat)
print(cm)
print(sum(diag(cm))/sum(cm))



##### neuralnet package ####### 

library(neuralnet)

df = data.frame(cbind(x,y))
nn = neuralnet(y~V1+V2+V3+V4+V5+V6+V7+V8+V9,data=df,hidden = 5)
yy = nn$net.result[[1]]
yhat = matrix(0,length(y),1)
yhat[which(yy > mean(yy))] = 1
yhat[which(yy <= mean(yy))] = 0
print(table(y,yhat))



###### h20 package install and creation of deep neural network ##### 

library(h2o)

### connection creation #####

localH2O = h2o.init(ip="localhost", port = 54321, 
                    startH2O = TRUE, nthreads=-1)

### training and test data import 

train <- h2o.importFile("BreastCancer.csv")
test <- h2o.importFile("BreastCancer.csv")

y_new = data.frame(y)


pd <- sample(2,nrow(df),replace = TRUE,prob = c(0.85,0.15))
train <- as.factor(df[pd==1,])
validate <- as.factor(df[pd==2,])



model = h2o.deeplearning(x=x, 
                         y=y, 
                         training_frame=train, 
                         validation_frame=test, 
                         distribution = "multinomial",
                         activation = "RectifierWithDropout",
                         hidden = c(10,10,10,10),
                         input_dropout_ratio = 0.2,
                         l1 = 1e-5,
                         epochs = 50)