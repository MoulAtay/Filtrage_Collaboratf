import streamlit as st
from login_page import show_login_page
from rating_page import show_rating_page
from recommendation_page import show_recommendation_page
from rate_more_page import show_rate_more_page
from rated_movies_page import show_rated_movies_page

# Initialisation de l'état de session
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "username" not in st.session_state:
    st.session_state["username"] = ""
if "initial_ratings_done" not in st.session_state:
    st.session_state["initial_ratings_done"] = False
if "menu" not in st.session_state:
    st.session_state["menu"] = "🏠 Recommandations"

# Barre latérale si connecté
if st.session_state["logged_in"] and st.session_state["initial_ratings_done"]:
    st.sidebar.title("📂 Navigation")
    st.session_state["menu"] = st.sidebar.radio(
        "Choisissez une page :", 
        ["🏠 Recommandations", "🎬 Noter plus de films", "📖 Mes films notés"]
    )

# Logique de navigation
if not st.session_state["logged_in"]:
    show_login_page()

elif not st.session_state["initial_ratings_done"]:
    show_rating_page()

else:
    if st.session_state["menu"] == "🏠 Recommandations":
        show_recommendation_page()
    elif st.session_state["menu"] == "🎬 Noter plus de films":
        show_rate_more_page()
    elif st.session_state["menu"] == "📖 Mes films notés":
        show_rated_movies_page()
