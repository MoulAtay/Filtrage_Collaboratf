import streamlit as st
import json
import os
from utils import get_next_ml_user_id

USER_FILE = "users.json"

# Charger les utilisateurs
def load_users():
    if os.path.exists(USER_FILE):
        with open(USER_FILE, "r") as f:
            return json.load(f)
    return {}

# Sauvegarder les utilisateurs
def save_users(users):
    with open(USER_FILE, "w") as f:
        json.dump(users, f, indent=4)

def show_login_page():
    users = load_users()
    mode = st.sidebar.selectbox("Choisissez une action", ["🔑 Connexion", "🆕 Inscription"])

    if mode == "🔑 Connexion":
        st.title("🎬 Movie Recommender - Connexion")
        username = st.text_input("Nom d'utilisateur")
        password = st.text_input("Mot de passe", type="password")

        if st.button("Se connecter"):
            if username in users and users[username]["password"] == password:
                st.success(f"Bienvenue {username} 👋")
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.session_state["user_id"] = users[username]["ml_user_id"]
                st.session_state["initial_ratings_done"] = users[username].get("initial_ratings_done", False)
            else:
                st.error("Nom d'utilisateur ou mot de passe incorrect.")

    elif mode == "🆕 Inscription":
        st.title("📝 Création de compte")
        new_username = st.text_input("Nom d'utilisateur")
        new_password = st.text_input("Mot de passe", type="password")

        if st.button("Créer le compte"):
            if new_username in users:
                st.warning("Ce nom d'utilisateur existe déjà.")
            elif new_username == "" or new_password == "":
                st.warning("Tous les champs sont obligatoires.")
            else:
                ml_user_id = get_next_ml_user_id()  # ← user_id = max + 1
                users[new_username] = {
                    "password": new_password,
                    "ml_user_id": int(ml_user_id)
                    # Pas besoin de user_idx ici, il sera généré avec LabelEncoder
                }
                save_users(users)
                st.success("Compte créé avec succès ✅ Vous pouvez maintenant vous connecter.")
