import streamlit as st
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
import joblib
import subprocess

@st.cache_data
def load_movies():
    return pd.read_csv("data/u.item", sep="|", encoding="latin-1", header=None, usecols=[0, 1], names=["movie_id", "title"])

@st.cache_data
def load_ratings():
    return pd.read_csv("data/new_ratings.csv") if os.path.exists("data/new_ratings.csv") else pd.DataFrame(columns=["user_id", "movie_id", "rating", "user_idx", "item_idx"])

def save_rating(user_id, movie_id, rating):
    import subprocess

    # Charger les anciennes notations
    ratings = load_ratings()

    # Charger les encodeurs
    user_enc = joblib.load("models/user_encoder.pkl")
    item_enc = joblib.load("models/item_encoder.pkl")

    try:
        # Essayer d'encoder normalement
        user_idx = user_enc.transform([user_id])[0]
        item_idx = item_enc.transform([movie_id])[0]
    except ValueError:
        # Mettre à jour les encodeurs si user_id ou movie_id inconnus
        new_user_ids = list(user_enc.classes_)
        if user_id not in new_user_ids:
            new_user_ids.append(user_id)
        user_enc.fit(new_user_ids)

        new_movie_ids = list(item_enc.classes_)
        if movie_id not in new_movie_ids:
            new_movie_ids.append(movie_id)
        item_enc.fit(new_movie_ids)

        # Refaire l'encodage
        user_idx = user_enc.transform([user_id])[0]
        item_idx = item_enc.transform([movie_id])[0]

        # Sauvegarder les encodeurs mis à jour
        joblib.dump(user_enc, "models/user_encoder.pkl")
        joblib.dump(item_enc, "models/item_encoder.pkl")

    # Supprimer les notations en doublon (même user_id et movie_id)
    ratings = ratings.drop(ratings[(ratings["user_id"] == user_id) & (ratings["movie_id"] == movie_id)].index)

    # Ajouter la nouvelle notation
    new_entry = pd.DataFrame([{
        "user_id": user_id,
        "movie_id": movie_id,
        "rating": rating,
        "user_idx": user_idx,
        "item_idx": item_idx
    }])
    ratings = pd.concat([ratings, new_entry], ignore_index=True)

    # Sauvegarder dans le fichier
    ratings.to_csv("data/new_ratings.csv", index=False)

    # 🔁 Réentraîner le modèle
    #with st.spinner("🔄 Mise à jour du modèle..."):
     #   subprocess.run(["python", "train_mlp_model.py"], check=True)

    st.success("🎉 Film noté avec succès !")


def show_rate_more_page():
    st.title("🍿 Noter d'autres films")

    movies = load_movies()
    search_query = st.text_input("Rechercher un film 🎬")

    if search_query:
        filtered_movies = movies[movies["title"].str.contains(search_query, case=False, na=False)].head(10)

        if filtered_movies.empty:
            st.warning("Aucun film trouvé.")
        else:
            for _, row in filtered_movies.iterrows():
                movie_id = row["movie_id"]
                title = row["title"]
                st.subheader(f"{title}")
                rating = st.slider(f"Note pour {title}", 1, 5, step=1, key=f"rating_{movie_id}")
                if st.button("Noter", key=f"btn_{movie_id}"):
                    user_id = st.session_state["user_id"]
                    save_rating(user_id, movie_id, rating)