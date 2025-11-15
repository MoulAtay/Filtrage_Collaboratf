import streamlit as st
import pandas as pd
import os
import csv
from sklearn.preprocessing import LabelEncoder
from login_page import load_users
from login_page import load_users, save_users


@st.cache_data
def load_data():
    ratings = pd.read_csv("data/u.data", sep="\t", names=["user_id", "movie_id", "rating", "timestamp"])
    movies = pd.read_csv("data/u.item", sep="|", encoding="latin-1", header=None, usecols=[0, 1], names=["movie_id", "title"])
    return ratings, movies

def save_user_ratings(user_id, user_ratings):
    # Charger les anciennes notations MovieLens
    ratings_df = pd.read_csv("data/u.data", sep="\t", names=["user_id", "movie_id", "rating", "timestamp"])
    
    # Charger les notations existantes si elles existent
    new_ratings_path = "data/new_ratings.csv"
    if os.path.exists(new_ratings_path):
        try:
            new_ratings_df = pd.read_csv(new_ratings_path)
        except pd.errors.EmptyDataError:
            new_ratings_df = pd.DataFrame(columns=["user_id", "movie_id", "rating"])

        ratings_df = pd.concat([ratings_df, new_ratings_df[["user_id", "movie_id", "rating"]]], ignore_index=True)

    # Ajouter les nouvelles notations de l'utilisateur
    new_entries = [{"user_id": user_id, "movie_id": int(mid), "rating": float(r)} for mid, r in user_ratings.items() if r > 0]
    new_df = pd.DataFrame(new_entries)
    ratings_df = pd.concat([ratings_df, new_df], ignore_index=True)

    # Encodage des user_id et movie_id
    user_enc = LabelEncoder()
    item_enc = LabelEncoder()
    ratings_df["user_idx"] = user_enc.fit_transform(ratings_df["user_id"])
    ratings_df["item_idx"] = item_enc.fit_transform(ratings_df["movie_id"])

    # Isoler les nouvelles notations avec leurs index encodés
    new_df = ratings_df[ratings_df["user_id"] == user_id][["user_id", "movie_id", "rating", "user_idx", "item_idx"]]

    # Enregistrer dans new_ratings.csv
    if os.path.exists(new_ratings_path):
        existing = pd.read_csv(new_ratings_path)
        combined = pd.concat([existing, new_df], ignore_index=True).drop_duplicates(subset=["user_id", "movie_id"])
    else:
        combined = new_df

    combined.to_csv(new_ratings_path, index=False)

def show_rating_page():
    st.title("🎥 Notation initiale")

    ratings, movies = load_data()

    # Top 20 films les plus populaires
    popular_movies = ratings['movie_id'].value_counts().reset_index()
    popular_movies.columns = ['movie_id', 'count']
    popular_movies = popular_movies.merge(movies, on='movie_id')
    top_movies = popular_movies.sort_values(by='count', ascending=False).head(20)

    if "user_ratings" not in st.session_state:
        st.session_state.user_ratings = {}

    st.subheader("Veuillez noter au moins 5 films pour continuer")

    for _, row in top_movies.iterrows():
        title = row['title']
        movie_id = row['movie_id']
        rating = st.slider(f"{title}", 0, 5, value=0, key=f"rating_{movie_id}")
        st.session_state.user_ratings[movie_id] = rating

    rated_count = sum(1 for r in st.session_state.user_ratings.values() if r >= 1)

    if rated_count >= 5:
        st.success(f"✅ {rated_count} films notés.")
        if st.button("Continuer vers les recommandations"):
            user_id = st.session_state["user_id"]
            save_user_ratings(user_id, st.session_state.user_ratings)
            st.session_state["initial_ratings_done"] = True
            # ✅ Mettre à jour users.json pour persister cette info
            users = load_users()
            username = st.session_state["username"]
            if username in users:
                users[username]["initial_ratings_done"] = True
                save_users(users)
            
    else:
        st.warning("⛔ Veuillez noter au moins 5 films.")
