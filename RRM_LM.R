# preprocess the Hitters data        
library(ISLR)                                                         #                                
fix(Hitters)                                                          # invoke edit on Hitters
names(Hitters)                                                        # show the predictors of Hitters 
dim(Hitters)                                                          # check how many predictors and observations in Hitters
sum(is.na(Hitters$Salary))                                            # find how many observations lacking of Salary data
Hitters = na.omit(Hitters)                                            # remove the observations lacking of Salary data 
dim(Hitters)                                                          # check again the fixed Hitters
sum(is.na(Hitters))                                                   #

# set observations and responce datasets
x = model.matrix(Salary~.,Hitters)[,-1]                               # produce the matrix of all abservation corresponding 19 predictors
# "[,-1]" means remove the first coloum from the original matrix
# additionally, the model.matrix can only take numerical, quantitative inputs
y = Hitters$Salary                                                    # set coloum Salary as responce

# The Ridge Regression Model (RRM)
# perform Ridge Regression for a range of lambda values
library(glmnet)                                                       #
grid = 10^seq(10,-2,length = 100)                                     # set range of lambda values

# make RRM
ridge.mod = glmnet(x,y,alpha = 0,lambda = grid)                       # Ridge Regression, argument alpha = 0 means ridge regression
plot(ridge.mod)                                                       # show the coefficients with L2-norm
dim(coef(ridge.mod))                                                  # check the dimention of coefficients matrix of RR model          

# check the results of the RRM above
ridge.mod$lambda[50]                                                  # find value of the 50th lambda
coef(ridge.mod)[,50]                                                  # check the coefficients with the 50th lambda

ridge.mod$lambda[60]                                                  # find value of the 60th lambda
coef(ridge.mod)[,60]                                                  # check the coefficients with the 60th lambda

predict(ridge.mod, s = 50, type = "coefficients")[1:20,]              # also, predict function can show the coefficients with any lambda within 
# its range
# s means the value of lambda to predict
# type = "coefficients" computes the coefficients for s
# [1:20,] show the results in row

# cross-validaion
# set a random train-datasets and test-datasets
set.seed(1)                                                           # set a results-reproducible random sample  
train = sample(1:nrow(x), nrow(x)/2)                                  # random set half of data as train datasets
test = (-train)                                                       # set the rest as test datasets
y.test = y[test]                                                      #

# check the results of CV
ridge.mod = glmnet(x[train,],y[train],alpha = 0,lambda = grid,thresh = 1e-12)   # train the RRM with train-dataset

ridge.pred = predict(ridge.mod,s = 4,newx = x[test,])                 # predict the test results with a small lambda value, s = 4    
# argument newx is matrix of new values to predict
mean((ridge.pred - y.test)^2)                                         # compute the MSE of test results

ridge.pred = predict(ridge.mod,s = 1e10,newx = x[test,])              # predict the test results with a large lambda value, s = 1e10                  
mean((ridge.pred - y.test)^2)                                         # compute the MSE of test results

ridge.pred = predict(ridge.mod,s = 0,newx = x[test,],exact = T)       # predict the test results just in the least squares (no penalty)
# argument exact is relevent only the lambda used to predict differd  
# from the used in the original model
mean((ridge.pred - y.test)^2)                                         # compute the MSE of test results

lm(y~x,subset = train)                                                # in general,the unpenalized least squares model
predict(ridge.mod,s = 0,exact = T,type = "coefficients")[1:20,]       # show the prediction results

# find the best lambda
set.seed(1)                                                           # recall the random sample data above
cv.out = cv.glmnet(x[train,],y[train],alpha = 0)                      # cv.glmnet perform a CV function with 10-fold CV,
# which can be also changed with argument nfolds
plot(cv.out)                                                          #  
bestlam = cv.out$lambda.min                                           # find the lambda making the minimal MSE
bestlam                                                               # show the best lambda 

# compute the optimal RRM with CV
ridge.pred = predict(ridge.mod,s = bestlam,newx = x[test,])           # predict the test results with the best lambda
mean((ridge.pred - y.test)^2)                                         # compute the MSE of test results

out = glmnet(x,y,alpha = 0)                                           # refit the full data to the RRM
ridge.coef = predict(out,type = "coefficients",s = bestlam)[1:20,]    # show the coefficients estimates result
ridge.coef

# The Lasso Model
# perform the Lasso Model for a range of lambda values
lasso.mod = glmnet(x[train,],y[train],alpha = 1)                      # Ridge Regression, argument alpha = 1 means Lasso
plot(lasso.mod)                                                       # show the coefficients with L1-norm
dim(coef(lasso.mod))                                                  # check the dimention of coefficients matrix of Lasso model

# find the best lambda
set.seed(1)                                                           # recall the random sample data above                   
cv.out = cv.glmnet(x[train,],y[train],alpha = 1)                      # cv.glmnet perform a CV function with 10-fold CV,
# which can be also changed with argument nfolds
plot(cv.out)                                                          #
bestlam = cv.out$lambda.min                                           # find the lambda making the minimal MSE
bestlam                                                               # show the best lambda

# compute the optimal RRM with CV
lasso.pred = predict(lasso.mod,s = bestlam,newx = x[test,])           # predict the test results with the best lambda
mean((lasso.pred - y.test)^2)                                         # compute the MSE of test results

out = glmnet(x,y,alpha = 1,lambda = grid)                             # refit the full data to the LM
lasso.coef = predict(out,type = "coefficients",s = bestlam)[1:20,]    # show the coefficients estimates result
lasso.coef