# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 17:03:56 2020

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
from collections import Counter
from imblearn.over_sampling import SMOTE

#conda install -c conda-forge textblob
#WORD EMBEDDING MODELS
from nltk import ngrams
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


# Modeling packages
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score

########################   data  cleaning ################################

amazon= pd.read_csv('E:\\datascience\\project\\excelr\\amazon review\\rev1.csv')

##converting the review to lower case
amazon['Reviews1']= amazon['Reviews'].apply(lambda x: " ".join(word.lower() for word in x.split()))

##removing the punctuations
amazon['Reviews1']=amazon['Reviews1'].apply(lambda x:''.join([i for i in x  if i not in string.punctuation]))

##removing the numericals
amazon['Reviews1']=amazon['Reviews1'].str.replace('[0-9]','')

##removing all the stopwords
stop_words=stopwords.words('english')
amazon['Reviews1']=amazon['Reviews1'].apply(lambda x: " ".join(word for word in x.split() if word not in stop_words))

###Applying Lemmitaization
amazon['Reviews1']= amazon['Reviews1'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

pattern = r"((?<=^)|(?<= )).((?=$)|(?= ))"
amazon['Reviews1']= amazon['Reviews1'].apply(lambda x:(re.sub(pattern, '',x).strip()))

###############################################################################

################################## sentimental analysis #######################
amazon2=amazon.copy()
amazon2['polarity'] = amazon2['Reviews1'].apply(lambda x: TextBlob(x).sentiment[0])

amazon2['sent'] = 0
amazon2.loc[amazon2['polarity'] > 0.2, 'sent'] = 1
amazon2.loc[amazon2['polarity'] <= 0.2, 'sent'] = -1

###########################             MODEL BUILDING              ##############################

####################################### Bag-of-words  ##################################

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



########################## Sampling #################


sm = SMOTE(random_state = 42) 
X_train_smo, y_train_smo = sm.fit_resample(X_train_bow, y_train_bow)

Counter(y_train_bow)
Counter(y_train_smo)

################################     training and testing       ####################

### 1. logistic regression

lr_model = LogisticRegression()

#Fit train and test into the model

lr_model.fit(X_train_smo, y_train_smo)

test_pred_smo = lr_model.predict(X_test_bow)
print("accuracy_score: ", f1_score(y_test_bow,test_pred_smo)) ### 91.03


#### 2. SVM

clf = LinearSVC()

#Fit train and test into the model
clf.fit(X_train_smo, y_train_smo)

#Predict the result
test_pred_smo = clf.predict(X_test_bow)
print("accuracy_score: ", f1_score(y_test_bow,test_pred_smo)) ##91.92

