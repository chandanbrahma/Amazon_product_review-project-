# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 19:19:42 2020

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

##########################        EDA       ############################################
amazon= pd.read_csv('E:\\datascience\\project\\excelr\\amazon review\\rev1.csv')
amazon.dtypes

amazon.isna().sum()


## checking the number of words per review
##getting the number of words by splitting them by a space
words_per_review = amazon.Reviews.apply(lambda x: len(x.split(" ")))
words_per_review.hist(bins = 100)
plt.xlabel('Review_length(words)')
plt.ylabel('Frequency')
plt.show()

words_per_review.mean() # ~47
words_per_review.skew() #5.51

# We can see that the no of words per review is highly positive with mean 47. Means the 
# means on avg review has 47 words

## NOW LETS LOOK AT THE DISTRIBUTION OF THE RATINGS
percentage_val= 100*amazon['Ratings'].value_counts()/len(amazon)

percentage_val

percentage_val.plot.bar()
plt.show()

##SO 60% OF THE RIVIEW ARE 5, 18% ARE 4, 6% ARE 3, 3% IS 2 AND 11% IS 1

##TEXT VISUALIZATION USING WORD CLOUD
OnePlus7T_string = " ".join(amazon['Reviews'])
wordcloud_ip = WordCloud(width=2400,height=1600,max_words=100).generate(OnePlus7T_string)
plt.imshow(wordcloud_ip)


data=[]
data=OnePlus7T_string.split()

import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
pos_word_list=[]
neu_word_list=[]
neg_word_list=[]

for word in data:
    if (sid.polarity_scores(word)['compound']) >= 0.5:
        pos_word_list.append(word)
    elif (sid.polarity_scores(word)['compound']) <= -0.5:
        neg_word_list.append(word)
    else:
        neu_word_list.append(word)       

##negative word cloud
OnePlus7T_string_neg = " ".join(neg_word_list)

wordcloud_neg_in_neg = WordCloud(
                      background_color='black',
                      width=2400,
                      height=1600,max_words=100
                     ).generate(OnePlus7T_string_neg)

plt.imshow(wordcloud_neg_in_neg)

##positive wordcloud
OnePlus7T_string_pos = " ".join(pos_word_list)
wordcloud_pos = WordCloud(
                      background_color='black',
                      width=2400,
                      height=1600,max_words=100
                     ).generate(OnePlus7T_string_pos)

plt.imshow(wordcloud_pos)



##################                DATA CLEANING              ###################
amazon.columns

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


##################              SENTIMENT ANALYSIS               #####################

# method1
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

sia = SIA()
results = []

for line in amazon['Reviews1']:
    pol_score = sia.polarity_scores(line)
    pol_score['Reviews1'] = line
    results.append(pol_score)
results

df = pd.DataFrame.from_records(results)
df.head
df['Sentimentss'] = 0
df.loc[df['compound'] > 0.5, 'Sentimentss'] = 1
df.loc[df['compound'] < -0.5, 'Sentimentss'] = -1
df.head(10)

amazon1=amazon.copy()
amazon1['sent']= df.Sentimentss
amazon1= amazon1[amazon1['sent'] != 0] ##3408 reviews
amazon1['sent'].value_counts().plot.bar()


# method2
amazon2=amazon.copy()
amazon2['polarity'] = amazon2['Reviews1'].apply(lambda x: TextBlob(x).sentiment[0])

amazon2['sent'] = 0
amazon2.loc[amazon2['polarity'] > 0.2, 'sent'] = 1
amazon2.loc[amazon2['polarity'] <= 0.2, 'sent'] = -1
# amazon2=amazon2[amazon2['sent'] != 0]   ##4603 reviews


amazon2['sent'].value_counts()

amazon2['sent'].value_counts().plot.bar()

#method 3
amazon3=amazon.copy()
amazon3['sent'] = 0
amazon3.loc[amazon3['Ratings'] > 3, 'sent'] = 1
amazon3.loc[amazon3['Ratings'] < 3, 'sent'] = -1
amazon3= amazon3[amazon3['sent'] != 0]  ##4675 reviews



amazon3['sent'].value_counts().plot.bar()

#METHOD4
df1=df.copy()
df1['Sentiments'] = 0
df1.loc[df1['pos'] >df1['neg'], 'Sentiments'] = 1
df1.loc[df1['neg']>df1['pos'],'Sentiments'] = -1
df1['Sentiments'].value_counts()

amazon4=amazon.copy()
amazon4['sent'] = df1['Sentiments']
amazon4= amazon4[amazon4['sent'] != 0] ##4642 reviews



amazon4.sent.value_counts().plot.bar()
