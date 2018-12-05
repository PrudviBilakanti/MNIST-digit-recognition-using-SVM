library(caret)
library(kernlab)
library(dplyr)
library(readr)
library(ggplot2)
library(gridExtra)

# For fast and parallel computing using all cores of the system .

library(doParallel)


detectCores()    # 4 cores

cl = makeCluster(4)
registerDoParallel(cl)

# Read the Train and Test digit csv files 




mnist_train <- read.csv("c:/practice/SVM Dataset/mnist_train.csv",header = F)
mnist_test <- read.csv("c:/practice/SVM Dataset/mnist_test.csv",header = F)


# Lets sample 15% of data from Train and Test data as suggested.

set.seed(100)

ind <- sample(1:nrow(mnist_train), 0.15*nrow(mnist_train))
testind <- sample(1:nrow(mnist_test), 0.15*nrow(mnist_test))

sample_mnist_train <- mnist_train[ind,]
sample_mnist_test <- mnist_test[testind,]

########################################################################################################################
#                                               Data Understanding and Preperation                                     #
########################################################################################################################

# Check the dimensions
dim(sample_mnist_train)
# Check the sructure of the dataset
str(sample_mnist_train)  # all numeric columns

# Check any missing values
sum(is.na(sample_mnist_train)) # 0 missing values
sum(is.na(sample_mnist_test))  # 0 missing values

# check any duplicate data in the dataset
sum(duplicated(sample_mnist_train))  # 0 duplicates
sum(duplicated(sample_mnist_test))    # 0 duplicates


# Change the traget dependent variable name in the train and test data
colnames(sample_mnist_train)[1] <- 'digit'
colnames(sample_mnist_test)[1] <- 'digit'

# COnvert the class label digit into factor

sample_mnist_train$digit <- as.factor(sample_mnist_train$digit)
sample_mnist_test$digit <- as.factor(sample_mnist_test$digit)

# we will scale the variables for easy computing 
# scale function doesnt work here as some columns has all 0's in it, with which when divided gives NaN values

# Lets divide the values with max values in the dataset

max(sample_mnist_train)  #// 255
max(sample_mnist_test)   #// 255

sample_mnist_train[,c(2:785)] <- sample_mnist_train[,c(2:785)]/255
sample_mnist_test[,c(2:785)] <- sample_mnist_test[,c(2:785)]/255


# Lets verify whether the class lables are balanced or not

table(sample_mnist_train[,1])
table(sample_mnist_test[,1])


# 0   1   2   3   4   5   6   7   8   9 
# 880 964 902 910 943 856 850 922 879 894  

# Data looks good in terms of label balance. so we can start modeling on the selected data.

ggplot(sample_mnist_train,aes(digit,fill=digit))+
  geom_bar()+
  labs(title='Digits count', 
       x='Digits',
       y='Count',
       fill="digit")+
  geom_text(aes(label=(..count..),vjust=-0.5),stat="count")



########################################################################################################################
#                                    Lets start model building - With default Parameters                               #
########################################################################################################################

#                                        Lets start with Linear SVM model.                                             #

Model_linear <- ksvm(digit~ ., data = sample_mnist_train, scale = FALSE, kernel = "vanilladot")

Eval_linear<- predict(Model_linear, sample_mnist_test)

confusionMatrix(Eval_linear, sample_mnist_test$digit)

#Overall Statistics
# Number of Support Vectors : 2554;  parameter : cost C = 1

# Accuracy : 0.926           
# 95% CI : (0.9116, 0.9387)

# Statistics by Class:
# 
#                      Class: 0 Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6 Class: 7 Class: 8 Class: 9
# Sensitivity           0.99310   0.9812   0.9172  0.91946  0.96324   0.8947   0.9000  0.94595  0.86275   0.8876
# Specificity           0.99114   0.9910   0.9955  0.98668  0.98680   0.9892   0.9955  0.99630  0.99183   0.9939





#                                   Degree 2 Polynomial Kernel with default parameters                                  #
#                                                                                                                       #

Model_poly <- ksvm(digit ~ ., data = sample_mnist_train, kernel = "polydot", scale = FALSE,kpar = list(degree = 2))

Eval_Poly <- predict(Model_poly, sample_mnist_test)

confusionMatrix(Eval_Poly, sample_mnist_test$digit)


# Number of Support Vectors : 2544 ; Hyperparameters : degree =  2  scale =  1 ; C = 1; 
# Accuracy : 0.9607          
# 95% CI : (0.9496, 0.9699)

# Statistics by Class:
#   
#                     Class: 0 Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6 Class: 7 Class: 8 Class: 9
#Sensitivity           1.00000   0.9812   0.9618  0.97315  0.96324   0.9737  0.9437  0.94595  0.95425   0.9213
#Specificity           0.99336   0.9955   0.9963  0.99704  0.99340   0.9957  0.9985  0.99704  0.99406   0.9955



#
#                                    RBF Kernel with default parameters                                                #
#                                 
Model_RBF <- ksvm(digit~ ., data = sample_mnist_train, scale = FALSE, kernel = "rbfdot")

Eval_RBF <- predict(Model_RBF, sample_mnist_test)

confusionMatrix(Eval_RBF, sample_mnist_test$digit)

# Number of Support Vectors : 3541; Hyperparameter : sigma =  0.0106305304851746

# Accuracy : 0.9567          
# 95% CI : (0.9451, 0.9664)

#                     Class: 0 Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6 Class: 7 Class: 8 Class: 9
#Sensitivity           1.00000   0.9875   0.9682  0.97315  0.95588  0.94737  0.9437  0.93243  0.94771   0.9157
#Specificity           0.99557   0.9948   0.9970  0.99556  0.99267  0.99495  0.9955  0.99704  0.99406   0.9947


# Comparision of above models

# With defualt parameters 2 degree Polynomial performs marginally better than linear and RBF kernels. Also it is clear
# that there exists some non linearity in the data as non linear models are performing better than linear model
# we will perform cross validation and find the optimal sigma and cost values to finalise the model which differentiates
# the classes.


########################################################################################################################
#                                 Hyperparameter tuning and Cross Validation                                           #
########################################################################################################################


# We will use the train function from caret package to perform crossvalidation
trainControl <- trainControl(method="cv", number=5)
# Number - Number of folds 
# Method - cross validation

metric <- "Accuracy"



#                                               Linear Kernel                                                          #


# making a grid of C values. 
grid <- expand.grid(C=seq(1, 5, by=1))

# Performing 5-fold cross validation
fit.svmlinear <- train(digit~., data=sample_mnist_train, method="svmLinear", metric=metric, tuneGrid=grid, trControl=trainControl)

# Printing cross validation result
print(fit.svmlinear)
#  C  Accuracy   Kappa    
#  1  0.9163366  0.9070178
#  2  0.9156696  0.9062764
#  3  0.9151142  0.9056593
#  4  0.9144474  0.9049182
#  5  0.9140029  0.9044243

#Accuracy was used to select the optimal model using the largest value.
#The final value used for the model was C = 1.


# Plotting "fit.svm" results
plot(fit.svmlinear)

# Lets the test the model on test data.

eval_linear <- predict(fit.svmlinear,sample_mnist_test)
confusionMatrix(eval_linear,sample_mnist_test$digit)

# Accuracy : 0.926 
#                     Class: 0 Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6 Class: 7 Class: 8 Class: 9
#Sensitivity           0.99310   0.9812   0.9172  0.91946  0.96324   0.8947 0.9000  0.94595  0.86275   0.8876
#Specificity           0.99114   0.9910   0.9955  0.98668  0.98680   0.9892 0.9955  0.99630  0.99183   0.9939



#
#                                           Non-Linear - Kernels - Polynomial                                          #
#


# making a grid of C, degree and scale values. 
Polygrid = expand.grid(C= c(0.01, 0.1, 1, 10), degree = c(1:3), scale = c(-10, -1, 1, 10))


# Performing 5-fold cross validation
polyfit.svm <- train(digit~., data=sample_mnist_train, method="svmPoly", metric=metric, tuneGrid=Polygrid, trControl=trainControl)

# Printing cross validation result
print(polyfit.svm)
# Best tune at C=0.01, degree=2 and scale 1 
# Accuracy - 0.960445536


# Plotting "fit.svm" results
plot(polyfit.svm)


# Tried with different parameters based on above results.
# Polygrid = expand.grid(C= seq(0.01, 0.05, by=0.01), degree = c(2:4), scale = c(1,4,by=1))

# The model's Accuracy is still high at C=0.01, degree =2 at scale =1
# as the degree of polynomial increased the model's accuracy decreased.




#                                   Non-Linear - Kernles - Radial Basis (RBF)                                         #
#

radialgrid <- expand.grid(.sigma=seq(0.01, 0.05, by=0.01), .C=seq(1, 5, by=1))


fit.svmradial <- train(digit~., data=sample_mnist_train, method="svmRadial", metric=metric, tuneGrid=radialgrid, trControl=trainControl,allowparallel=T)
plot(fit.svmradial)
print(fit.svmradial)

# Best Value among all the folds
#Sigma = 0.03   C=3  Accuracy = 0.9698896 


eval_rbf_new <- predict(fit.svmradial,sample_mnist_test)
confusionMatrix(eval_rbf_new,sample_mnist_test$digit)


#Accuracy : 0.968     

#                       Class: 0 Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6 Class: 7 Class: 8 Class: 9
# Sensitivity           1.00000   0.9875   0.9745  0.99329  0.97059  0.97368   0.9625  0.93243  0.96732   0.9270
# Specificity           0.99631   0.9955   0.9985  0.99630  0.99413  0.99711   0.9985  0.99852  0.99480   0.9947

# Accuracy on test data is same as training data and looks good. Sensitivity and Specificity metrics also looks 
# great for each individual classes.


# Tried with different hyper parameters as mentioned below.
# radialgrid <- expand.grid(.sigma=seq(0.03, 0.04, by=0.001), .C=seq(3, 5, by=0.5))

# The Accuracy was 0.9704 at sigma = 0.032 and C=5, but when validates on test data the Accuracy was similar
# to previous model. Though the model accuracy is high, and prediction on tested data is same, considering
# previous model, keeping generilasation in mind.

######################################################## Conclusion ####################################################

# After examining the results of all models with and without parameters, there seems to be some non linearity
# in the data and Radial Kernel performed much better with 0.968 Accuaracy when compared to all other linear and non linear models.

# Stop Cluster to release all cores.
stopCluster(cl)

