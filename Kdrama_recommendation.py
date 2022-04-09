#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np 
import pandas as pd 
import seaborn as sns


# In[4]:


# from google.colab import drive
# drive.mount('/content/drive')


# In[5]:


drama = pd.read_csv('top100_kdrama.csv')


# In[6]:


drama.head()


# In[7]:


drama['Name'].describe()


# In[8]:


start = drama


# In[9]:


# genre,rank,tags,name,synopsis,cast
drama.info()


# In[10]:


drama = drama[['Name','Genre','Rank','Tags','Synopsis','Cast']]


# In[11]:


drama.head()


# In[12]:


drama.isnull().sum()


# In[13]:


drama['Cast'][0]


# In[14]:


drama['Tags'][0]


# In[15]:


# test="test string".split() 
# "_".join(test)


# In[16]:


def collapse(L):
    L1 = []
    test1 =" ".join(L)
    L1.append(test1)
    return L1


# In[17]:


drama['Cast'] = drama['Cast'].apply(collapse)
drama['Tags'] = drama['Tags'].apply(collapse)
drama['Genre'] = drama['Genre'].apply(collapse)


# In[18]:


drama.head()


# In[19]:


# import ast
# def convert(text):
#     L = []
#     for i in ast.literal_eval(text):
#         L.append(i['name']) 
#     return L 


# In[20]:


# drama['Genre'] = drama['Genre'].apply(convert)
# drama.head()


# In[21]:


drama['Cast'] = drama['Cast'].apply(lambda x:[i.replace(" ","") for i in x])
drama['Tags'] = drama['Tags'].apply(lambda x:[i.replace(" ","") for i in x])
drama['Genre'] = drama['Genre'].apply(lambda x:[i.replace(" ","") for i in x])


# In[22]:


drama.head()


# In[23]:


drama.iloc[0].Tags


# In[24]:


drama['Synopsis'] = drama['Synopsis'].apply(lambda x:x.split())


# In[25]:


drama['mix'] = drama['Synopsis'] + drama['Genre'] + drama['Tags'] + drama['Cast']


# In[26]:


new_drama = drama[['Rank','Name','mix']]


# In[27]:


new_drama.head()


# In[28]:


new_drama['mix'][0]


# In[29]:


new = drama.drop(columns=['Synopsis','Genre','Tags','Cast'])


# In[30]:


new.head()


# In[31]:


new['mix'] = new['mix'].apply(lambda x: " ".join(x))
new.head()


# In[32]:


new['mix'][0]


# In[33]:


new['mix'] = new['mix'].apply(lambda x:x.lower())


# In[34]:


new['Name'] = new['Name'].apply(lambda x:x.casefold())


# In[35]:


new.head()


# In[36]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')


# In[37]:


vector = cv.fit_transform(new['mix']).toarray()


# In[38]:


vector[0]


# In[39]:


cv.get_feature_names()


# In[40]:


get_ipython().system('pip install nltk')


# In[41]:


import nltk


# In[42]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[43]:


def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


# In[44]:


new['mix'] = new['mix'].apply(stem)


# In[45]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')


# In[46]:


vector = cv.fit_transform(new['mix']).toarray()


# In[47]:


cv.get_feature_names()


# In[48]:


vector.shape


# In[49]:


from sklearn.metrics.pairwise import cosine_similarity


# In[50]:


similarity = cosine_similarity(vector)


# In[51]:


similarity.shape


# In[52]:


similarity[0]


# In[ ]:





# In[53]:


def recommend(movie):
    movie = movie.casefold()
    if movie not in new['Name'].unique():
        print("This Kdrama is not in our database")
    else:
        index = new[new['Name'] == movie].index[0]
        distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
        for i in distances[1:6]:
            print(new.iloc[i[0]].Name)


# In[ ]:





# In[54]:


new


# In[55]:


new['Name'][30]


# In[56]:


recommend("It's Okay to Not Be Okay")


# In[57]:


#DRAMA EXISTS IN THE LIST OR NOT
new[new['Name'].str.contains('squid game')]


# In[58]:


start.head()


# In[59]:


start['Rating'].nlargest(n=5)


# In[60]:


start['Rating'].hist(bins=10)


# In[61]:


import pickle
pickle.dump(new, open('movies.pkl','wb'))


# In[64]:


new['Name'].values


# In[ ]:




