# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 18:38:46 2020

@author: Chandan
"""
import pandas as pd
import numpy as np
import requests   
from bs4 import BeautifulSoup as bs 
import re
import string 
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk import word_tokenize
import matplotlib.pyplot as plt
from wordcloud import WordCloud
nltk.download('wordnet')
from nltk.corpus import wordnet
import textblob
from textblob import TextBlob
from textblob import Word

#conda install -c conda-forge textblob
#WORD EMBEDDING MODELS
from nltk import ngrams
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


# Modeling packages
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score,confusion_matrix,accuracy_score

##STANDARDIZATION OF RATINGS
## RATING 4 & 5 WILL BE MAPPED TO 1 AND WILL BE REFERED TO POSITIVE REVIEW
## RATING 1 & 2 WILL BE MAPPED TO 0 AND WILL BE REFERED TO NEGATIVE REVIEW
## RATING 3  ARE REMOVED AS THEY ARE NEUTRAL

# amazon['sentiment_rating'] = np.where(amazon['Ratings'] > 3,1,0)

# amazon= amazon[amazon['Ratings'] != 3] ##4675 reviews



# ## NO OF POSITIVE AND NEGATIVE SENTIMATES
# amazon['sentiment_rating'].value_counts()
# #1    3926
# #0     749

# amazon['sentiment_rating'].value_counts().plot.bar()




########################           PREPROCESSING            ################################
# amazon['new_reviews'] = amazon['Reviews'].str.lower()


### 1.USING WORD TOKENIZE LETS TRY TO EXTRACT THE INDIVISUAL WORDS
# token_list_lower = [ word_tokenize(each) for each in amazon['new_reviews']]
# tokens_lower =[item for sublist in token_list_lower for item in sublist]
# len(tokens_lower) #227672


# ## 2.REMOVING THE SPECIAL CHARACTER 
# special_character= amazon['new_reviews'].apply(lambda review :[char for char in list(review) if not  char.isalnum() and char != ' '])

# #getting the list of list in a single list
# all_list = [item for sublist in special_character for item in sublist]
# set(all_list)

##now lets remove all those special characters
# amazon_backup= amazon['new_reviews'].copy()
# amazon['new_reviews'] = amazon['new_reviews'].replace (r'[^A-Za-z0-9]+'," " )

# ip_rev_string = re.sub("[^A-Za-z" "]+"," ",ip_rev_string).lower()
# ip_rev_string = re.sub("[0-9" "]+"," ",ip_rev_string)

# ## 3. REMOVING STOPWORDS
# eng_stop_words = stopwords.words('english')

# stop_words= set(eng_stop_words)
# noise_words=[]
# without_stop_words= []
# stopword= []
# sentence= amazon['new_reviews'][1]
# words= nltk.word_tokenize(sentence)

# for word in words :
#     if word in stop_words:
#         stopword.append(word)
#     else:
#         without_stop_words.append(word)

# def stopwords_removal(stop_words, sentence):
#     return [word for word in nltk.word_tokenize(sentence) if word not in stop_words]
# amazon['reviews_text_nonstop'] =  amazon['new_reviews'].apply(lambda row:' ' .join(stopwords_removal(stop_words, row) ))
# amazon['reviews_text_nonstop']
      
            





###########################             MODEL BUILDING              ##############################

## 1. Bag-of-words

vec = CountVectorizer()
X = vec.fit_transform(amazon2['Reviews1'])
df = pd.DataFrame(X.toarray(), columns = vec.get_feature_names())
df.head()

##Let's use this to create a bag of words from the reviews, excluding the noise words we identified earlier:
noise_words=[]
bow_counts = CountVectorizer(tokenizer= word_tokenize, # type of tokenization
                             stop_words=noise_words, # List of stopwords
                             ngram_range=(1,1)) # number of n-grams
 
bow_data = bow_counts.fit_transform(amazon2['Reviews1'])


##Once the bag of words is prepared, the dataset should be divided into training and test sets:

X = amazon2['Reviews1']
y=amazon2['sent']
X_train_bow, X_test_bow, y_train_bow, y_test_bow = train_test_split(bow_data, # Features
                                                                    amazon2['sent'], # Target variable
                                                                    test_size = 0.2, # 20% test size
                                                                    random_state = 0) # random state for replication purposes

##Applying logistic regression to BOW features
lr_model = LogisticRegression()
lr_model = RandomForestClassifier()


lr_model.fit(X_train_bow, y_train_bow)

# Predicting the results
test_pred_lr = lr_model.predict(X_test_bow)

print("accuracy_score: ", f1_score(y_test_bow,test_pred_lr))

#Amazon1=98.04
#amazon2=96.34 (0) //// 92.76 (0.2)
#amazon3=94.23
#amazon4=95.12

##We can even get interpretable features from this in terms of what contributed the most to positive and negative sentiment:
lr_weights = pd.DataFrame(list(zip(bow_counts.get_feature_names(), # ge tall the n-gram feature names
                                   lr_model.coef_[0])), # get the logistic regression coefficients
                          columns= ['words','weights']) # defining the colunm names

lr_weights.sort_values(['weights'], ascending = False)[:10] # top-15 more important features for positive reviews


## 2. TF-IDF model

### Creating a python object of the class CountVectorizer
tfidf_counts = TfidfVectorizer(tokenizer= word_tokenize, # type of tokenization
                               stop_words=noise_words, # List of stopwords
                               ngram_range=(1,1)) # number of n-grams

tfidf_data = tfidf_counts.fit_transform(amazon2['Reviews1'])

## train test split

X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(tfidf_data,
                                                                            amazon2['sent'],
                                                                            test_size = 0.2,
                                                                            random_state = 0)

print("Positive: ", X_train_tfidf.label.value_counts()[0]/len(train)*100,"%")

##Applying logistic regression to TF-IDF features

### Setting up the model class
lr_model_tf_idf = LogisticRegression()

## Training the model 
lr_model_tf_idf.fit(X_train_tfidf,y_train_tfidf)

## Prediciting the results
test_pred_lr2 = lr_model_tf_idf.predict(X_test_tfidf)

## Evaluating the model
print("F1 score: ",f1_score(y_test_tfidf, test_pred_lr2)) ##90.06

#amazon1=96.47
#amazon2=94.22 (0)//// 90.79(0.2)
#amazon3=93.66
#amazon4=94.81


# y_train_bow['sent'].value_count()
# pd.Series(y_train_bow, name='sent').nunique()


from collections import Counter
Counter(y_train_bow)
Counter(y_train_smo)

from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 42) 
X_train_smo, y_train_smo = sm.fit_resample(X_train_bow, y_train_bow)
lr_model = LogisticRegression()
lr_model.fit(X_train_smo, y_train_smo)

test_pred_smo = lr_model.predict(X_test_bow)
print(classification_report(y_test_bow,test_pred_smo))
print("accuracy_score: ", f1_score(y_test_bow,test_pred_smo)) ### 91.03

