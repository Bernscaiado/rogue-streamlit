import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

st.title('Rogue Inteligent Movie Recomendation systems')


@st.cache
def get_dataframes():
    df = pd.read_csv('out.csv')

    return df

@st.cache
def get_random_subset():
    dt = pd.read_csv('path.csv')
    return dt.sample(n=5)

random_subset = get_random_subset()

key = 0
ratings = []
movies = []

for movie in random_subset.title:
    key += 1
    st.write(movie)
    movies.append(movie)
    ratings.append(st.number_input('Give a Rating', key=key, min_value=1,max_value=5))

def standardize(row):
    new_row = (row - row.mean()) / (row.max() - row.min())
    return new_row


def get_similar_movies(movie_name,user_rating):
    similar_score = item_similarity_df[movie_name]*(user_rating-2.5)
    similar_score = similar_score.sort_values(ascending=False)
    return similar_score

similar_movies = pd.DataFrame()

if st.button('SUBMIT'):
    df = get_dataframes()
    ratings_std = df.apply(standardize)
    item_similarity = cosine_similarity(ratings_std.T)
    item_similarity_df = pd.DataFrame(item_similarity, index=df.columns, columns=df.columns)

    data_set = np.array((movies,ratings)).T

    for movie,rating in data_set:
        similar_movies =  similar_movies.append(get_similar_movies(movie,int(rating)),ignore_index=True)

    st.write(similar_movies.sum().sort_values(ascending=False))
