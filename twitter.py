import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import csv
import pickle
from sklearn.svm import LinearSVC
import preprocessor as p
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer

df=pd.read_csv('train 8k.csv')
df = df.sample(frac=1).reset_index(drop=True)
df['label'].value_counts()
df['hashtag'] = df['tweet'].apply(lambda x: re.findall(r"#(\w+)", x))
df['hashtag'] = [' '.join(map(str, l)) for l in df['hashtag']];

with open('stopwords_hinglish.csv', newline='') as f:
    reader = csv.reader(f)
    hinglish = list(reader)
    
hinglish = [''.join(ele) for ele in hinglish] 

def clean_tweets(df):
  tempArr = []
  for line in df:
    #send to tweet_processor
    tmpL = p.clean(line)
    
    tempArr.append(tmpL)
  return tempArr

df['tweet']=clean_tweets(df['tweet']);
df['tweet'] = df['tweet'].str.cat(df['hashtag'], sep =" ")

def preprocess(sms):
  pattern = re.compile('[^[A-Za-z\s]')
  sms = re.sub(pattern,' ',sms)
  sms = re.sub('\s+',' ',sms)
  sms = sms.lower()
  return sms

df['tweet']= df['tweet'].map(lambda x :preprocess(x))

from nltk.corpus import stopwords
stop = stopwords.words('english')
stop.extend(hinglish)
df['tweet'] = df['tweet'].apply(lambda x: ' '.join([item for item in x.split() if item not in stop]))

y=df['label']
nlp=df['tweet']
X_train, X_val, y_train, y_val = train_test_split(nlp, y, test_size = 0.25, random_state = 0)

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)

pickle.dump(vectorizer,open('transform.pkl','wb'))

X_val = vectorizer.transform(X_val)

classifier = LinearSVC()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_val)

print(metrics.confusion_matrix(y_val, y_pred))
print(metrics.classification_report(y_val, y_pred, digits=3))
print(accuracy_score(y_val, y_pred))

filename='nlp_model.pkl'
pickle.dump(classifier,open(filename,'wb'))

freqs = [(word, X_train.getcol(idx).sum()) for word, idx in vectorizer.vocabulary_.items()]

def Convert(tup, di): 
    di = dict(tup) 
    return di     
dic = {} 
dic=Convert(freqs, dic)

w = WordCloud(width=1600,height=1200,mode='RGBA',background_color='white',max_words=50).fit_words(dic)
plt.imshow(w)

