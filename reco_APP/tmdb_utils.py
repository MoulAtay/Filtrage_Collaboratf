import requests
import os

TMDB_API_KEY = "c369d81884c2220f96b0175ea53b8f87"
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

# Dossier cache pour éviter de faire trop d'appels API
CACHE_FILE = "data/poster_cache.json"

import json
def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

def get_movie_poster(title):
    cache = load_cache()
    if title in cache:
        return cache[title]

    params = {
        "api_key": TMDB_API_KEY,
        "query": title
    }
    response = requests.get(f"{TMDB_BASE_URL}/search/movie", params=params)

    if response.status_code == 200:
        results = response.json().get("results")
        if results and results[0].get("poster_path"):
            poster_url = TMDB_IMAGE_BASE + results[0]["poster_path"]
            cache[title] = poster_url
            save_cache(cache)
            return poster_url

    cache[title] = None
    save_cache(cache)
    return None
