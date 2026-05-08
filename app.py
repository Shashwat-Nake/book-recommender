import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load the Slim Data
@st.cache_data
def load_data():
    books = pd.read_csv("books_slim.csv")
    ratings = pd.read_csv("ratings_slim.csv")
    return books, ratings

books, ratings = load_data()

# 2. Build Pivot Table (On-the-fly)
pt = ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')
pt.fillna(0, inplace=True)
similarity_scores = cosine_similarity(pt)

def recommend(book_name):
    index = np.where(pt.index == book_name)[0][0]
    distances = similarity_scores[index]
    similar_items = sorted(list(enumerate(distances)), key=lambda x: x[1], reverse=True)[1:6]
    
    data = []
    for i in similar_items:
        data.append(pt.index[i[0]])
    return data

# 3. UI
st.title("📚 Book Recommendation System")
user_input = st.selectbox("Select a Book:", pt.index.tolist())

if st.button('Recommend'):
    try:
        res = recommend(user_input)
        for i in res:
            st.write(f"📖 {i}")
    except:
        st.error("Error finding recommendations.")
