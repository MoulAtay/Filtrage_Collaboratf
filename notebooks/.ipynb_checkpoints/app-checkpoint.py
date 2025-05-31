# app.py - Recommendation System Interface
import tensorflow as tf
import numpy as np
import pickle
import os

# Define RatingScaler class (must match training)
class RatingScaler(tf.keras.layers.Layer):
    def __init__(self, min_rating, max_rating, **kwargs):
        super().__init__(**kwargs)
        self.min_rating = min_rating
        self.max_rating = max_rating
        
    def call(self, inputs):
        return inputs * (self.max_rating - self.min_rating) + self.min_rating
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'min_rating': self.min_rating,
            'max_rating': self.max_rating
        })
        return config

class RecommendationSystem:
    def __init__(self, model_dir='saved_models'):
        self.model_dir = model_dir
        self.models = {}
        self.mappings = {}
        self.movie_titles = {}
        self.loaded_model = None
        
        # Load movie titles
        self.load_movie_titles()
        
        # Load mappings and available models
        self.load_resources()
    
    def load_movie_titles(self):
        """Load movie title database"""
        try:
            movies = pd.read_csv('../data/movies_clean.csv')
            self.movie_titles = dict(zip(movies['movie_id'], movies['title']))
        except Exception as e:
            print(f"Error loading movie titles: {str(e)}")
            self.movie_titles = {}
    
    def load_resources(self):
        """Load all necessary resources"""
        # Load mappings
        mappings_path = os.path.join(self.model_dir, 'mappings.pkl')
        if os.path.exists(mappings_path):
            with open(mappings_path, 'rb') as f:
                self.mappings = pickle.load(f)
        
        # Discover available models
        self.available_models = {}
        for file in os.listdir(self.model_dir):
            if file.endswith('.keras'):
                model_name = file.replace('_model.keras', '')
                self.available_models[model_name] = os.path.join(self.model_dir, file)
        
        print(f"Available models: {list(self.available_models.keys())}")
    
    def load_model(self, model_name):
        """Load a specific model into memory"""
        if model_name not in self.available_models:
            print(f"Model '{model_name}' not found!")
            return False
        
        try:
            self.loaded_model = tf.keras.models.load_model(
                self.available_models[model_name],
                custom_objects={'RatingScaler': RatingScaler}
            )
            print(f"Loaded {model_name} model successfully")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            self.loaded_model = None
            return False
    
    def get_recommendations(self, user_id, top_n=10):
        """Generate recommendations for a user"""
        if not self.loaded_model:
            print("No model loaded! Please load a model first.")
            return []
        
        if user_id not in self.mappings.get('user_id_to_idx', {}):
            print(f"User {user_id} not found in database")
            return []
        
        # Prepare inputs
        user_idx = self.mappings['user_id_to_idx'][user_id]
        movie_ids = list(self.mappings['movie_id_to_idx'].keys())
        movie_indices = list(self.mappings['movie_id_to_idx'].values())
        
        # Create input arrays
        user_indices = np.full(len(movie_ids), user_idx)
        movie_indices = np.array(movie_indices)
        
        # Predict ratings
        predictions = self.loaded_model.predict(
            [user_indices, movie_indices], 
            verbose=0
        ).flatten()
        
        # Clip to valid rating range
        min_rating = self.mappings.get('min_rating', 0.5)
        max_rating = self.mappings.get('max_rating', 5.0)
        predictions = np.clip(predictions, min_rating, max_rating)
        
        # Get top recommendations
        top_indices = np.argsort(predictions)[::-1][:top_n]
        return [(movie_ids[i], predictions[i]) for i in top_indices]
    
    def display_recommendations(self, recommendations):
        """Display recommendations in a user-friendly format"""
        if not recommendations:
            print("No recommendations available")
            return
        
        print("\nTop Recommendations:")
        for i, (movie_id, rating) in enumerate(recommendations, 1):
            title = self.movie_titles.get(movie_id, f"Movie ID {movie_id}")
            print(f"{i}. {title} - Predicted Rating: {rating:.2f}")

def main():
    # Initialize recommendation system
    recommender = RecommendationSystem()
    
    print("\n" + "="*50)
    print("MOVIE RECOMMENDATION SYSTEM")
    print("="*50)
    
    while True:
        # Model selection
        print("\nAvailable models:")
        for i, model_name in enumerate(recommender.available_models.keys(), 1):
            print(f"{i}. {model_name}")
        print("0. Exit")
        
        choice = input("\nSelect a model (number): ")
        
        # Exit condition
        if choice == '0':
            print("Exiting...")
            break
        
        # Validate selection
        try:
            choice_idx = int(choice) - 1
            model_names = list(recommender.available_models.keys())
            if choice_idx < 0 or choice_idx >= len(model_names):
                print("Invalid selection!")
                continue
            
            model_name = model_names[choice_idx]
            if not recommender.load_model(model_name):
                continue
        except ValueError:
            print("Please enter a valid number")
            continue
        
        # Get user ID
        while True:
            user_id = input("\nEnter user ID (or 'back' to select another model): ")
            if user_id.lower() == 'back':
                break
            
            try:
                user_id = int(user_id)
                recommendations = recommender.get_recommendations(user_id)
                recommender.display_recommendations(recommendations)
            except ValueError:
                print("Please enter a valid user ID (integer)")
            except Exception as e:
                print(f"Error generating recommendations: {str(e)}")

if __name__ == "__main__":
    main()