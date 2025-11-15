# Image de base avec Python
FROM python:3.10-slim

# Crée un dossier dans la boîte
WORKDIR /app

# Mets tous tes fichiers dans la boîte
COPY . .

# Mets à jour pip + installe les bibliothèques
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Ouvre la porte 8501 (celle de Streamlit)
EXPOSE 8501

# Lancer Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]
