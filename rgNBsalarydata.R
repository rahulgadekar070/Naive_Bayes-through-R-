###Prepare a classification model using "Naive Bayes"###

#Problem-2 :- Salary Data_Train & Test...

salary_data_train <-  read.csv("C:/Users/admin/Desktop/RG_Assgn_data/16. SVM/SalaryData_Train(1).csv")
View(salary_data_train)
#so here "salary" is our output class.
str(salary_data_train)
table(salary_data_train$Salary)    #<=50k-22653 & >50k-7508

#Remove Space problem for levels in train dataset in salary (o/p) column...
salary_data_train[,14] <- trimws(salary_data_train[,14])
salary_data_train[,14] <- as.factor(salary_data_train[,14])   #convert into factor


salary_data_test <- read.csv("C:/Users/admin/Desktop/RG_Assgn_data/16. SVM/SalaryData_Test(1).csv")
View(salary_data_test)
str(salary_data_test)
table(salary_data_test$Salary)    #<=50k-11360 & >50k-3700

levels(salary_data_train$Salary)
levels(salary_data_test$Salary)

### EDA ###
sum(is.na(salary_data_train))
sum(is.na(salary_data_test))

#Always check variables are in factor or not.
salary_data_train$educationno <- as.factor(salary_data_train$educationno)
salary_data_test$educationno  <- as.factor(salary_data_test$educationno)

library(tm)

prop.table(table(salary_data_train$Salary))
prop.table(table(salary_data_test$Salary))

#Buld model on train & test data
library(e1071)
salary_train_model <- naiveBayes(salary_data_train$Salary~., data = salary_data_train)
salary_train_model
pred_train   <- predict(salary_train_model,newdata=salary_data_test)
pred_train

table <- table(pred_train,salary_data_test$Salary)
table
library(gmodels)
CrossTable(pred_train,salary_data_test$Salary)

#try another model by "laplace"
salary_train_model2 <- naiveBayes(salary_data_train$Salary~., data = salary_data_train,laplace = 4)
salary_train_model2
pred_train2 <- predict(salary_train_model2,salary_data_test)
pred_train2

table2 <- table(pred_train2,salary_data_test$Salary)
table2

CrossTable(pred_train2,salary_data_test$Salary)

#Accuracy...
Accuracy <- sum(diag(table))/sum(table)
Accuracy       #81.87%

#Accutacy2...
Accuracy2 <- (sum(diag(table2))/sum(table2))
Accuracy2      #81.83%

#Inferences :- from model1 it conclude that, there are less wrong predictions compared to 2nd model.
#so the Model1 has better accuracy than model2. 
