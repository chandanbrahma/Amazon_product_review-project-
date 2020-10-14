# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 00:15:40 2020

@author: Chandan
"""

import requests   
from bs4 import BeautifulSoup as bs 
import re 
import nltk
from nltk.corpus import stopwords

import matplotlib.pyplot as plt
from wordcloud import WordCloud

OnePlus7T_reviews =[]
OnePlus7T_reviews_date=[]
OnePlus7T_ratings=[]


OnePlus7T_reviews =[]
OnePlus7T_reviews_date=[]
OnePlus7T_ratings=[]

for i in range(1,500):
  ip=[]  
  dp=[]
  rate = []
  url = 'https://www.amazon.in/Test-Exclusive-748/product-reviews/B07DJLVJ5M/ref=cm_cr_arp_d_paging_btm_next_501?ie=UTF8&reviewerType=all_reviews&pageNumber='+str(i)
  response = requests.get(url)
  soup = bs(response.content,"html.parser")# creating soup object to iterate over the extracted content 
  reviews = soup.findAll("span",attrs={"class","a-size-base review-text review-text-content"})
  reviews_date = soup.findAll("span",attrs={"class","a-size-base a-color-secondary review-date"})
  rating = soup.find_all('i',class_='review-rating')
  
  # Extracting the content under specific tags  
  for i in range(len(reviews)):
    reviews[i].text.replace("\n"," ")
    ip.append(reviews[i].text)
    dp.append(reviews_date[i].text)
    rate.append(rating[i].text)
  OnePlus7T_reviews=OnePlus7T_reviews+ip  # adding the reviews of one page to empty list which in future contains all the reviews
  OnePlus7T_reviews_date=OnePlus7T_reviews_date+dp
  OnePlus7T_ratings=OnePlus7T_ratings+rate

OnePlus7T_reviews
len(OnePlus7T_reviews)
OnePlus7T_reviews = [x.replace('\n','') for x in OnePlus7T_reviews]

OnePlus7T_reviews
OnePlus7T_ratings
OnePlus7T_reviews_date


df = pd.DataFrame()
df['Ratings']=OnePlus7T_ratings
df['Reviews']=OnePlus7T_reviews
df['Dates']=OnePlus7T_reviews_date
df.head()
len(df)
df.to_csv(r'E:\datascience\project\excelr\amazon review\reviews.csv',index=True)
df.columns

df['Dates']= df['Dates'].str.strip('Reviewed in India on')

df['Ratings']=df['Ratings'].replace('out of 5 stars', '', regex=True)
df['Ratings']=df['Ratings'].astype(float)
df['Ratings']=df['Ratings'].astype(int)

df['Dates']=pd.to_datetime(df['Dates'])




