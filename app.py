import streamlit as st
import pickle
import pandas as pd
st.title('Kdrama Recommender System')

st.beta_set_page_config(page_title='Kdrama-Addicts | Kdrama Recommender')

def recommend(movie):
        movie = movie.casefold()
        index = movies[movies['Name'] == movie].index[0]
        distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
        recommended_movies = []
        for i in distances[1:6]:
            recommended_movies.append(movies.iloc[i[0]].Name)
        return recommended_movies

movies_dict = pickle.load(open('movies_dict.pkl','rb'))
movies  = pd.DataFrame(movies_dict)

similarity = pickle.load(open('similarity.pkl','rb'))

# for i in movies['Name']:
m = movies['Name'].values
option = st.selectbox(
  'Pick your Kdrama',
  m)

if st.button('Recommend'):
    recommendations = recommend(option)
    for i in recommendations:
        st.write(i)
