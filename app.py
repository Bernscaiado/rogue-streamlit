import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

image_size = st.slider('Zoom', 50, 250, 119)


TEACHERS = {
    'Bruno Lajoie' : {"url":'https://avatars1.githubusercontent.com/u/22095643?v=4', "rating": 0},
    'Kevin Robert' : {"url":'https://avatars1.githubusercontent.com/u/9978111?v=4', "rating": 0},
    'Filme3' : {"url":'https://avatars1.githubusercontent.com/u/22095643?v=4', "rating": 0},
    'Filme4' : {"url":'https://avatars1.githubusercontent.com/u/9978111?v=4', "rating": 0},
    'Filme5' : {"url":'https://avatars1.githubusercontent.com/u/22095643?v=4', "rating": 0},
    'Filme6' : {"url":'https://avatars1.githubusercontent.com/u/9978111?v=4', "rating": 0},
}
TEACHER_CSS = f"""
#teachers {{
    display: flex;
    flex-wrap: wrap;
}}
.teacher-card {{
    display: flex;
    flex-direction: column;
}}
.teacher-card img {{
    width: {image_size}px;
    height: {image_size}px;
    border-radius: 100%;
    padding: 4px;
    margin: 10px;
    box-shadow: 0 0 4px #eee;
}}
.teacher-card span {{
    text-align: center;
}}
"""
# streamlit html injection seems to sensitive to new lines...
# remove that \ and the html gets displayed instead of being interpreted
TEACHER_CARD = """\
    <div class="teacher-card">
        <img src="{url}">
        <span>{name}</span>
        <span>{rating}</span>
    </div>
"""
teachers = ''.join([TEACHER_CARD.format(name=f'{name.split()[0]}', url=info["url"], rating=st.sidebar.number_input(name,1,5)) for name, info in TEACHERS.items()])
TEACHER_HTML = f"""
<style>
{TEACHER_CSS}
</style>
<div id="teachers">
    {teachers}
</div>
"""
st.write(TEACHER_HTML, unsafe_allow_html=True)

st.title('Rogue Inteligent Movie Recomendation systems')

dt = pd.read_csv('path.csv')
df = pd.read_csv('df.csv')
random_subset = dt.head(3)

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


ratings_std = df.apply(standardize)
item_similarity = cosine_similarity(ratings_std.T)
item_similarity_df = pd.DataFrame(item_similarity, index=df.columns, columns=df.columns)


def get_similar_movies(movie_name,user_rating):
    similar_score = item_similarity_df[movie_name]*(user_rating-2.5)
    similar_score = similar_score.sort_values(ascending=False)
    return similar_score

similar_movies = pd.DataFrame()

if st.button('SUBMIT'):
    data_set = np.array((movies,ratings)).T

    for movie,rating in data_set:
        similar_movies =  similar_movies.append(get_similar_movies(movie,int(rating)),ignore_index=True)

    st.write(similar_movies.sum().sort_values(ascending=False))
