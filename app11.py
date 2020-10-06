# -*- coding: utf-8 -*-
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
import pandas as pd
import datetime
import pickle
from flask import Markup
import numpy as np
from nltk import ngrams
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from textblob import TextBlob
from sklearn.model_selection import train_test_split
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk import word_tokenize
from matplotlib import pyplot as plt
import requests
from bs4 import BeautifulSoup
import io
import random
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


# Load the regression model 
filename = 'model.pkl'
regpredict = pickle.load(open(filename, 'rb'))


app11 = Flask(__name__,template_folder='templates')
 
@app11.route("/")
def home():    
    return render_template('DAQQ.html')

@app11.route("/", methods=['GET', 'POST'])
def predict():    
    if request.method == 'POST':
        datevalue=request.form['ddate']
        date = datetime.datetime.strptime(datevalue, '%Y-%m-%d')
        # Get the input from post request
        Prod_link=request.form['prod']
        header = {'User-Agent': 'Mozilla'}
        def Search(search_query):
            url="https://www.amazon.in/s?k="+search_query 
            page=requests.get(url,headers=header)  
            if page.status_code==200:
                return page                                
            else:
                return "Error"
        def url_link(query):
            url="https://www.amazon.in/dp/"+query     #search link
            page=requests.get(url,headers=header)
            if page.status_code==200:
                return page                            #return the page if there is no error
            else:
                return "Error"
        def content(query):
            url="https://www.amazon.in/"+query
            page=requests.get(url,headers=header)
            if page.status_code==200:
                return page
            else:
                return "Error"
        reviews=[]
            #ratings=[]
        datess=[]
        for i in range(1,501):                               #each product , it scrapes 500 pages of reviews
            cont_response=content(str(Prod_link)+str(i))    #iterates through multiple pages of the reviews
            soup=BeautifulSoup(cont_response.content)
            for j in soup.findAll("span",attrs={'data-hook':'review-body'}):
                reviews.append(j.text)                        #saves review content
 #           for star in soup.findAll("i",attrs={'data-hook':"review-star-rating"}):
#                ratings.append(star.text) 
            for d in soup.findAll("span",attrs={'data-hook':'review-date'}):
                datess.append(d.text)
        reviews[:] = [reviews.lstrip('\n\n') for reviews in reviews]
        reviews[:] = [reviews.lstrip('\n\n') for reviews in reviews]
        df1 = pd.DataFrame()
           #df1['Ratings']=ratings
        df1['Reviews']=reviews
        df1['Rev_date']=datess
           #df1['Ratings']=df1['Ratings'].replace('out of 5 stars', '', regex=True)
        df1['Rev_date']=df1['Rev_date'].replace('Reviewed in India on','', regex=True)
           #data['Ratings']=data['Ratings'].astype(float)
        #df1['Rev_date']=datetime.datetime.strptime(df1['Rev_date'], '%Y-%m-%d').date()
        df1['Rev_date']=pd.to_datetime(df1['Rev_date'])
        df1=df1.sort_values(by='Rev_date')
        
        
        def prep(text):
            corpus = []
            for i in range(0, len(text)):
                review = re.sub('[^a-zA-Z]', ' ', text[i])
                review = review.lower()
                review = review.split()
                lmtzr = WordNetLemmatizer()
                all_stopwords = stopwords.words('english')
                wor=("oneplus","phone","mobile","one","plus","amazon")
                review = [lmtzr.lemmatize(word) for word in review if not word in wor if not word in set(all_stopwords)]
                review = ' '.join(review)
                corpus.append(review)
            return corpus
        df1['cleaned_data']=prep(df1['Reviews'])
        df1=df1.set_index('Rev_date')
        def sentiment(text):
            polarity=[]
            subjectivity=[]
            for i in text.values:
                try:
                    analysis=TextBlob(i)
                    polarity.append(analysis.sentiment.polarity)
                    subjectivity.append(analysis.sentiment.subjectivity)
                except:
                    polarity.append(0)
                    subjectivity.append(0)
            df1['Polarity']=polarity
            df1['Subjectivity']=subjectivity
            df1['Senti_text'] = 0
            df1.loc[df1['Polarity'] > 0.2, 'Senti_text'] = 1
            df1.loc[df1['Polarity'] <= 0.2, 'Senti_text'] = -1
        sentiment(df1['cleaned_data'])
        #df1.to_csv('dataf.csv')
        bow_counts = CountVectorizer(tokenizer= word_tokenize)
        bow_data = bow_counts.fit_transform(df1['cleaned_data'].values.astype('U'))
        df3 = pd.DataFrame(bow_data.toarray(), columns = bow_counts.get_feature_names(),index=df1.index)
        X_test=df3[df3.index > date]
        X_train=df3[df3.index <= date]
        y_train=df1[df1.index <= date]['Senti_text']
        y_test=df1[df1.index > date]['Senti_text']
        lr_model = LogisticRegression()
        lr_model.fit(X_train, y_train)

        test_predr = lr_model.predict(X_test)
        
        output = 'The overall sentiment for test reviews is Pos.' if np.mean(test_predr) > 0.2 else 'The overall sentiment for test reviews is Neg'
        
        y_train.resample('w').mean().plot()
        y_test.resample('w').mean().plot()
        #plt.show()
        #plt.savefig('C:\\Users\\POONG POONG\\Desktop\\Project_ExcelR\\New_project\\myplot3.png')
        out='The mean is ' +str(np.mean(test_predr))
        
     
        
        y_train.resample('w').mean().plot()
        y_test.resample('w').mean().plot()
        from io import BytesIO
        figfile = BytesIO()
        plt.savefig(figfile, format='png')
        figfile.seek(0)  # rewind to beginning of file
        import base64
            #figdata_png = base64.b64encode(figfile.read())
        figdata_png = base64.b64encode(figfile.getvalue())
        #return figdata_png
        
        return render_template('DAQQ.html',prediction=output,plots=out,result=figdata_png.decode('utf8'))
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        #vec = CountVectorizer()
        #X = vec.fit_transform(df1['Reviews'])
        #df = pd.DataFrame(X.toarray(), columns = vec.get_feature_names())
        
        #X_train_bow, X_test_bow, y_train_bow, y_test_bow = train_test_split(X, # Features
         #                                                           df1['Senti_text'], # Target variable
         #                                                           test_size = 0.2, # 20% test size
         #                                                           random_state = 0)
        #bow_counts = CountVectorizer(tokenizer= word_tokenize,ngram_range=(1,1))
        #bow_data = bow_counts.fit_transform(df1['Reviews'])
        #regpredict1 = LogisticRegression()
        #regpredict1.fit(X_train_bow, y_train_bow)
        #predi=regpredict1.predict(X_test_bow)
        #output=list(prediction)  
        #output=np.mean(predi)
            #return output#print(output)
        #return render_template('DAQQ.html',prediction=output)

            #eng_stop_words = stopwords.words('english')
            #stop_words= set(eng_stop_words)
            #noise_words=[]
          #without_stop_words= []
          #stopword= []
          #vec = CountVectorizer()
          #X = vec.fit_transform(df1['Reviews'])
          #df = pd.DataFrame(X.toarray(), columns = vec.get_feature_names())
          

        
        
         
        
if __name__ == '__main__':  
   app11.run(debug = True) 
   
   
   
   
   
