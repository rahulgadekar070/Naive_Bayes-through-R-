###Prepare a classification model using "Naive Bayes"###

#Problem-1 :- SMS Raw Data 

sms_data <- read.csv("C:/Users/admin/Desktop/sms_raw_NB.csv")
View(sms_data)
#here "Type" is our output data. that whether it is "ham" or "spam".
class(sms_data)
str(sms_data)
table(sms_data$type)    #our problem is classificstion problem ao always take "Table".
#our gien TYpe data is in factor so we dont need to convert it into "factor", but if it is in "char" then convert it into "factor".
#sms_data$type <- factor(sms_data$type)  #here data always in "factor"
#str(sms_data) 

#Import the "Text_Mining" library "tm"
library(tm)

#convert the "text" Data into vector source that is "corpus"
sms_corpus <- Corpus(VectorSource(sms_data$text))
sms_corpus$content[1:10]    #from text data take first 10 rows by content functon.
#but in this data there are some numbers, punctuations & this is unnecessary data...
#so we want to remove it.

#cleaning data (remove unwanted symbols/data)
corpus_clean <- tm_map(sms_corpus,tolower)   #tM-map is function & it gives order that convert all data into "lowercase"
corpus_clean <- tm_map(corpus_clean,removeNumbers)
corpus_clean <- tm_map(corpus_clean,removeWords, stopwords)  #here remove stopwords.
corpus_clean <- tm_map(corpus_clean,removePunctuation)
removeNumPunct <- function(x) gsub("[^[:alpha:][:space:]]*", "", x ) #substitute the all inbetween "" this into blank data as "".
corpus_clean <- tm_map(corpus_clean,content_transformer(removeNumPunct))
corpus_clean <- tm_map(corpus_clean,stripWhitespace)   #remove whitespace
class(corpus_clean)

corpus_clean$content[1:10]

#create Document Term Matrix(DTM) for this clean data...in DTM-Docs in rows & Terms in cols.
sms_dtm <- DocumentTermMatrix(corpus_clean)
class(sms_dtm)     #simple_triplex_matrix

#creating train & test datasets...
sms_raw_train <- sms_data[1:4169,]
sms_raw_test  <- sms_data[4170:5559,]

#train & test for dtm data...
sms_dtm_train <- sms_dtm[1:4169,]
sms_dtm_test  <- sms_dtm[4170:5559,]

#train & test for corpus data...
corpus_clean_train <- corpus_clean[1:4169]     #here remember "," not taken
corpus_clean_test  <- corpus_clean[4170:5559]  #here remember "," not taken

#check that proportion of spam is similar
prop.table(table(sms_raw_train$type))
prop.table(table(sms_raw_test$type))

#indicator features for frequent words...
sms_dict <- findFreqTerms(sms_dtm_train,5)   #indicate the word that repeated 5 times atleast.
sms_dict        #so there are 1000 words are used frequently for 5 times.
#list(sms_dict[1:100])

#check this dict data for train & test
sms_train <- DocumentTermMatrix(corpus_clean_train,list(dictionary=sms_dict))
sms_test  <- DocumentTermMatrix(corpus_clean_test,list(dictionary=sms_dict))

#convert counts into factors...
convert_counts <- function(x) {
  x <- ifelse(x > 0,1,0)         #if specific word use for more that 0 times then it mention "1" & if not, mention "0".
  x <- factor(x, levels = c(0,1), labels = c("No","Yes"))    #here wherever "0" then=No & for "1"=Yes
} 

#apply this convert_counts function to columns of train/test data...
sms_train <- apply(sms_train, MARGIN = 2,convert_counts)
sms_test  <- apply(sms_test, MARGIN = 2,convert_counts)

#Training a model on the data...
library(e1071)
sms_classifier <- naiveBayes(sms_train,sms_raw_train$type)
sms_classifier

#Evaluating model performance by prediction...
sms_test_pred <- predict(sms_classifier,sms_test)
sms_test_pred
table <- table(sms_test_pred,sms_raw_test$type)
table(sms_raw_test$type)
#so from this model there 1202-ham & 156-spam rightly predicted but 5-ham & 27-spam predicted wrong.
#so build another model to reduce this wrong predictions.

library(gmodels)  #this is for model performance
CrossTable(sms_test_pred,sms_raw_test$type,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted','actual'))

#try another model by "laplace"
sms_classifier2 <- naiveBayes(sms_train,sms_raw_train$type,laplace = 4)  #change laplace value for better model.
sms_classifier2      #this laplace is R & D method.
sms_test_pred2  <- predict(sms_classifier2,sms_test)
table1 <- table(sms_test_pred2,sms_raw_test$type)
table1        #here ham-5 & spam-47 wrong predictions so this is not best fit model.

#Accuracy...
Accuacy1 <- (sum(diag(table))/sum(table))
Accuacy1          #97.70%

Accuracy2 <- (sum(diag(table1))/sum(table1))
Accuracy2         #96.25%

#Inferences :- from model1 it conclude that, there are less wrong predictions compared to 2nd model.
#so the Model1 has better accuracy than model2. 
