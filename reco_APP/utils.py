import pandas as pd
import os, json

def get_next_ml_user_id(filepath="data/u.data", user_file="users.json"):
    try:
        # Lire le fichier de ratings MovieLens pour connaître l’ID max existant
        ratings = pd.read_csv(filepath, sep="\t", names=["user_id", "movie_id", "rating", "timestamp"])
        max_existing_id = int(ratings["user_id"].max())
    except Exception as e:
        # En cas d'erreur de lecture, on suppose 943 utilisateurs existants (MovieLens 100k)
        max_existing_id = 943

    # Calculer l’ID max en incluant les utilisateurs déjà enregistrés dans users.json
    max_id = max_existing_id
    if os.path.exists(user_file):
        with open(user_file, "r") as f:
            users_data = json.load(f) or {}
            if users_data:
                # Trouver l’identifiant le plus élevé parmi les ml_user_id existants
                max_user_id = max(info.get("ml_user_id", 0) for info in users_data.values())
                if max_user_id > max_id:
                    max_id = max_user_id

    # Le prochain nouvel ID utilisateur sera le max existant + 1
    return max_id + 1