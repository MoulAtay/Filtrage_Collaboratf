import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dot, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import subprocess

# 📥 1. Charger les données
df_base = pd.read_csv("data/u.data", sep="\t", names=["user_id", "movie_id", "rating", "timestamp"])
df_base = df_base[["user_id", "movie_id", "rating"]]

# Vérifier si des nouvelles notations existent
if os.path.exists("data/new_ratings.csv"):
    df_new = pd.read_csv("data/new_ratings.csv")[["user_id", "movie_id", "rating"]]
    df = pd.concat([df_base, df_new], ignore_index=True)
else:
    df = df_base

# 🔁 2. Encoder les ID
user_enc = LabelEncoder()
item_enc = LabelEncoder()
df["user_idx"] = user_enc.fit_transform(df["user_id"])
df["item_idx"] = item_enc.fit_transform(df["movie_id"])

num_users = df["user_idx"].nunique()
num_items = df["item_idx"].nunique()

print(f"Training MLP model on {len(df)} ratings | {num_users} users | {num_items} items")

from tensorflow.keras.layers import Flatten, Dense, Dropout, Concatenate

embedding_size = 50

user_input = Input(shape=(1,), name="user_input")
item_input = Input(shape=(1,), name="item_input")

user_embedding = Embedding(input_dim=num_users, output_dim=embedding_size)(user_input)
item_embedding = Embedding(input_dim=num_items, output_dim=embedding_size)(item_input)

user_vec = Flatten()(user_embedding)
item_vec = Flatten()(item_embedding)

concat = Concatenate()([user_vec, item_vec])

x = Dense(128, activation="relu")(concat)
x = Dropout(0.2)(x)
x = Dense(64, activation="relu")(x)
x = Dropout(0.2)(x)

output = Dense(1)(x)

model = Model(inputs=[user_input, item_input], outputs=output)

model.compile(optimizer=Adam(0.001), loss=MeanSquaredError(), metrics=["mae"])

# 🧠 4. Entraîner
X_user = df["user_idx"].values
X_item = df["item_idx"].values
y = df["rating"].values

model.fit([X_user, X_item], y, batch_size=64, epochs=5, verbose=1)

# 💾 5. Sauvegarder
os.makedirs("models", exist_ok=True)
model.save("models/mlp_model.keras")

print("✅ Modèle MLP sauvegardé dans models/mlp_model.keras")


import joblib

joblib.dump(user_enc, "models/user_encoder.pkl")
joblib.dump(item_enc, "models/item_encoder.pkl")
