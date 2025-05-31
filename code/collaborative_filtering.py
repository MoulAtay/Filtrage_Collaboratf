import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error

class CollaborativeFiltering:
    def __init__(self, data_path='../data/ratings_clean.csv'):
        self.ratings = pd.read_csv(data_path)
        self.train_data, self.test_data = self._split_data()
        self.user_item_matrix = None
        self.item_user_matrix = None
        self.user_similarity = None
        self.item_similarity = None

    def _split_data(self):
        """Split data into train/test sets with stratification"""
        return train_test_split(
            self.ratings,
            test_size=0.2,
            stratify=self.ratings['user_id'],
            random_state=42
        )

    def _create_matrices(self):
        """Create user-item and item-user matrices"""
        self.user_item_matrix = self.train_data.pivot_table(
            index='user_id',
            columns='movie_id',
            values='rating'
        ).fillna(0)
        self.item_user_matrix = self.user_item_matrix.T

    def _calculate_similarities(self):
        """Pre-compute similarity matrices"""
        self.user_similarity = cosine_similarity(self.user_item_matrix)
        self.item_similarity = cosine_similarity(self.item_user_matrix)

    def user_based_predict(self, user_id, movie_id, k=10):
        """User-based prediction with fallback strategies"""
        try:
            if user_id not in self.user_item_matrix.index:
                return np.nan
            
            similar_users = np.argsort(self.user_similarity[user_id-1])[::-1][1:k+1]
            similar_ratings = self.user_item_matrix.iloc[similar_users][movie_id]
            valid_ratings = similar_ratings[similar_ratings != 0]
            
            if len(valid_ratings) > 0:
                return valid_ratings.mean()
            
            user_avg = self.user_item_matrix.loc[user_id].mean()
            if not np.isnan(user_avg):
                return user_avg
            
            return self.train_data['rating'].mean()
        except KeyError:
            return np.nan

    def item_based_predict(self, user_id, movie_id, k=10):
        """Item-based prediction with fallback strategies"""
        try:
            if movie_id not in self.item_user_matrix.index:
                return np.nan
            
            movie_idx = self.item_user_matrix.index.get_loc(movie_id)
            similar_items = np.argsort(self.item_similarity[movie_idx])[::-1][1:k+1]
            similar_movie_ids = self.item_user_matrix.index[similar_items]
            similar_ratings = self.user_item_matrix.loc[user_id, similar_movie_ids]
            valid_ratings = similar_ratings[similar_ratings != 0]
            
            if len(valid_ratings) > 0:
                return valid_ratings.mean()
            
            item_avg = self.item_user_matrix.loc[movie_id].mean()
            if not np.isnan(item_avg):
                return item_avg
            
            user_avg = self.user_item_matrix.loc[user_id].mean()
            if not np.isnan(user_avg):
                return user_avg
            
            return self.train_data['rating'].mean()
        except KeyError:
            return np.nan

    def evaluate(self, model_type='user', k=10):
        """Evaluate model performance"""
        self._create_matrices()
        self._calculate_similarities()
        
        valid_test_data = self.test_data[
            (self.test_data['user_id'].isin(self.user_item_matrix.index)) &
            (self.test_data['movie_id'].isin(self.user_item_matrix.columns))
        ]
        
        actual = []
        predicted = []
        predict_func = self.user_based_predict if model_type == 'user' else self.item_based_predict
        
        for _, row in valid_test_data.iterrows():
            pred = predict_func(row['user_id'], row['movie_id'], k)
            if not np.isnan(pred):
                actual.append(row['rating'])
                predicted.append(pred)
        
        if len(actual) > 0:
            rmse = np.sqrt(mean_squared_error(actual, predicted))
            mae = mean_absolute_error(actual, predicted)
            print(f"\n{model_type.title()}-Based Collaborative Filtering")
            print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}")
            print("Sample Predictions:")
            for i in range(min(5, len(actual))):
                print(f"User {valid_test_data.iloc[i]['user_id']} -> Movie {valid_test_data.iloc[i]['movie_id']}: "
                      f"Predicted {predicted[i]:.1f} vs Actual {actual[i]}")
        else:
            print("Error: No valid predictions generated")

# Usage Example
if __name__ == "__main__":
    cf = CollaborativeFiltering()
    
    print("Evaluating User-Based CF:")
    cf.evaluate(model_type='user', k=15)  # Try different k values
    
    print("\nEvaluating Item-Based CF:")
    cf.evaluate(model_type='item', k=15)