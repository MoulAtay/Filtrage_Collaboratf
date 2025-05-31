import pandas as pd

def load_and_clean_data():
    ratings = pd.read_csv('../data/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
    movies = pd.read_csv('../data/u.item', sep='|', encoding='latin-1', 
                        names=['movie_id', 'title', 'release_date', 'video_release', 'imdb_url', 
                               'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 
                               'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 
                               'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 
                               'Sci-Fi', 'Thriller', 'War', 'Western'])
    
    # Drop unnecessary columns
    movies.drop(columns=['video_release', 'imdb_url', 'unknown'], inplace=True)
    ratings.drop(columns=['timestamp'], inplace=True)
    
    # Cleaning steps
    movies.drop(columns=['video_release', 'imdb_url', 'unknown'], inplace=True)
    ratings.drop(columns=['timestamp'], inplace=True)
    
    # Save cleaned data
    ratings.to_csv('data/ratings_clean.csv', index=False)
    movies.to_csv('data/movies_clean.csv', index=False)

if __name__ == "__main__":
    load_and_clean_data()