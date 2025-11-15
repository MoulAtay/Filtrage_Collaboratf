import streamlit as st
import pandas as pd

@st.cache_data
def load_movies():
    return pd.read_csv("data/u.item", sep="|", encoding="latin-1", header=None, usecols=[0, 1], names=["movie_id", "title"])

@st.cache_data
def load_rated_movies():
    try:
        return pd.read_csv("data/new_ratings.csv")
    except FileNotFoundError:
        return pd.DataFrame(columns=["user_id", "movie_id", "rating", "user_idx", "item_idx"])

def show_rated_movies_page():
    st.title("📖 Vos films déjà notés")

    user_id = st.session_state["user_id"]
    ratings = load_rated_movies()
    movies = load_movies()

    # Filtrer les films notés par cet utilisateur
    user_ratings = ratings[ratings["user_id"] == user_id]

    if user_ratings.empty:
        st.info("Vous n'avez encore noté aucun film.")
        return

    # Joindre avec les titres
    user_ratings = user_ratings.merge(movies, on="movie_id", how="left")

    # Afficher sous forme de tableau
    for _, row in user_ratings.iterrows():
        st.write(f"🎬 **{row['title']}** — Note : ⭐ {row['rating']}/5")
