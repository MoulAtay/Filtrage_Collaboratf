import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import subprocess
import joblib
from tmdb_utils import get_movie_poster


@st.cache_resource
def load_mlp_model(path="models/mlp_model.keras"):
    return load_model(path)

@st.cache_data
def load_ratings_and_movies():
    import joblib

    # Charger u.data
    df_base = pd.read_csv("data/u.data", sep="\t", names=["user_id", "movie_id", "rating", "timestamp"])
    df_base = df_base[["user_id", "movie_id", "rating"]]

    # Charger les nouvelles notations
    df_new = pd.read_csv("data/new_ratings.csv")[["user_id", "movie_id", "rating"]]
    ratings = pd.concat([df_base, df_new], ignore_index=True)

    # Charger les encodages identiques à ceux du modèle
    user_enc = joblib.load("models/user_encoder.pkl")
    item_enc = joblib.load("models/item_encoder.pkl")

    ratings["user_idx"] = user_enc.transform(ratings["user_id"])
    ratings["item_idx"] = item_enc.transform(ratings["movie_id"])

    # Charger les titres de films
    movies = pd.read_csv("data/u.item", sep="|", encoding="latin-1", header=None, usecols=[0, 1], names=["movie_id", "title"])

    return ratings, movies


def get_user_predictions(model, user_id, ratings, movies, top_n=10):
    user_idx = ratings[ratings["user_id"] == user_id]["user_idx"].iloc[0]

    # Films déjà notés
    rated_movies = ratings[ratings["user_id"] == user_id]["movie_id"].unique()

    # 🔁 Prendre tous les films présents dans le dataset ratings (ceux encodés)
    all_movies = ratings["movie_id"].unique()
    unrated_movies = np.setdiff1d(all_movies, rated_movies)

    # Mapping ID → item_idx (encodé)
    movie_id_to_idx = ratings[["movie_id", "item_idx"]].drop_duplicates().set_index("movie_id")["item_idx"].to_dict()
    movie_id_to_title = movies.set_index("movie_id")["title"].to_dict()

    # Créer les couples (movie_id, item_idx) seulement pour ceux qu’on connaît
    candidates = [(mid, movie_id_to_idx[mid]) for mid in unrated_movies if mid in movie_id_to_idx]

    if not candidates:
        return []

    X_user = np.array([user_idx] * len(candidates)).reshape(-1, 1)
    X_item = np.array([item_idx for _, item_idx in candidates]).reshape(-1, 1)

    predictions = model.predict([X_user, X_item], verbose=0).flatten()
    top_indices = predictions.argsort()[-top_n:][::-1]

    # Exclure les films déjà notés (au cas où le modèle les recommande quand même)
    rated_movie_ids = ratings[ratings["user_id"] == user_id]["movie_id"].astype(int).tolist()

    top_movie_ids = []
    top_titles = []

    results = []
    for i in top_indices:
        movie_id = candidates[i][0]
        if movie_id not in rated_movie_ids:
            title = movie_id_to_title.get(movie_id, f"Film {movie_id}")
            score = float(f"{predictions[i]:.2f}")
            results.append((title, score))
        if len(results) >= top_n:
            break

    return results

def show_recommendation_page():
    st.title("🎯 Vos recommandations personnalisées")

    top_n = st.selectbox("Combien de films veux-tu ?", [5, 10, 15])

    if st.button("Générer mes recommandations"):
        user_id = st.session_state["user_id"]

        try:
            # 🔁 (Ré)entraîner le modèle à chaque clic
            with st.spinner("🔄 Entraînement du modèle MLP en cours..."):
                subprocess.run(["python", "train_mlp_model.py"], check=True)
                st.cache_data.clear()
                st.cache_resource.clear()

            # ✅ Charger le modèle mis à jour
            model = load_mlp_model()
            ratings, movies = load_ratings_and_movies()

            recommendations = get_user_predictions(model, user_id, ratings, movies, top_n)
            if recommendations:
                username = st.session_state.get("username", "utilisateur")
                st.success(f"Voici vos recommandations personnalisées, {username} 🎬")
                for i, (title, score) in enumerate(recommendations, 1):
                    st.markdown(f"**{i}. {title} — Note prévue : ⭐ {score}/5**")
                    poster_url = get_movie_poster(title)
                    if poster_url:
                        st.image(poster_url, width=150)
            else:
                st.warning("Aucune recommandation possible : tous les films sont déjà notés.")
        except Exception as e:
            st.error(f"Erreur pendant la prédiction : {e}")