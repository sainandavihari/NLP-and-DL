import requests
from bs4 import BeautifulSoup
import re, string,unicodedata
import nltk
from nltk import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer,LancasterStemmer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Asking server to get the info of the web page
r=requests.get(r'https://www.britannica.com/plant/tree')


soup=BeautifulSoup(r.content,'html.parser')

def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]','',text)
                  
def denoise_text():
    text=soup.get_text()
    text=remove_between_square_brackets(text)
    text=re.sub(' ','',text)
    return text
        

sample=denoise_text()
print(sample)
corpus=[]
corpus1=[]
sample=sample.lower()
corpus=word_tokenize(sample)
stop_words=set(stopwords.words('english'))
for i in range(len(corpus)):
    t=re.sub('[^a-zA-Z]',' ',corpus[i])
    if t not in stop_words:
           ps=PorterStemmer()
           t=ps.stem(t)
           corpus1.append(t)
    
len(corpus1)

corpus1[3]

wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(corpus1))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


