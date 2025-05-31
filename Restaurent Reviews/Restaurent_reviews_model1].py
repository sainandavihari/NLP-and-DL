import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv(r'/Users/sainandaviharim/Downloads/5th,6th  - NLP project/4.CUSTOMERS REVIEW DATASET/Restaurant_Reviews.tsv',delimiter = '\t', quoting = 3)

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split

corpus=[]

for i in range(0,100):
    review=re.sub('[^a-zA-Z]',' ',data['Review'][i])
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    #review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=[ps.stem(word) for word in review]
    review=' '.join(review)
    corpus.append(review)

#converting text to numbers using embeddings model

# applying count vectorizer and tfidf vectorizer

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

# implementiong tfidf vectorizer 
'''from sklearn.feature_extraction.text import TfidfVectorizer
cv=TfidfVectorizer()'''
x=cv.fit_transform(corpus).toarray()
y= data.iloc[0:100,1].values

# splitting data into train and test data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

# appplying decision tree ml algo

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)

y_pred=dt.predict(x_test)

from sklearn.metrics import classification_report,accuracy_score
cr=classification_report(y_test, y_pred)
print(cr)

acc=accuracy_score(y_test, y_pred)
print(acc)


# Accuracy of model is 65%

bias= dt.score(x_train,y_train)
var=dt.score(x_test,y_test)
print(bias,var)

# high bias and low variance whhich shows that the model is underfitted

#  using tfidf vectorizer, accracy is 50% , bias =100, var=s0
# using count vectorizer, accuracy is 65%, bias=100,var=55



# After including stop words, using decison tree ml algo , accursacy is 60% , bias=100,var=60

# applying lgbm ml alkgo
import lightgbm as lgbm
lgbm1=lgbm.LGBMClassifier()
lgbm1.fit(x_train,y_train)
y_pred1=lgbm1.predict(x_test)

from sklearn.metrics import classification_report,accuracy_score
cr=classification_report(y_test, y_pred1)
print(cr)

acc=accuracy_score(y_test, y_pred1)
print(acc)



# Accuracy of model is 65%

bias= lgbm1.score(x_train,y_train)
var=lgbm1.score(x_test,y_test)
print(bias,var)

# using lgbm , accurqcy is 50% and  bias=100,var=55

#  After including stop words, using lgbm classifier ml, accuracy is 50% , bias =61,var=50


# So best model is decision tree ml algorithm with stop words using count vectorizer.



