from flask import Flask, render_template, request
from googleapiclient.discovery import build
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Set up the YouTube API client
api_key = 'AIzaSyAElImp-8XtnQemYcMpvhRkXKtQYmpVECU'  # Replace with your actual API key
youtube = build('youtube', 'v3', developerKey=api_key)

# Load movie data and perform necessary preprocessing
movies_data = pd.read_csv('D:\\flask_projects\\flask_movie_video\\movies.csv')
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')
combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)
similarity = cosine_similarity(feature_vectors)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        movie_name = request.form['movie_name']
        recommended_movies = recommend_movies(movie_name)
        video_data = get_related_videos(recommended_movies)
        return render_template('index.html', movie_name=movie_name, recommended_movies=video_data)
    return render_template('index.html')

def recommend_movies(movie_name):
    find_close_match = difflib.get_close_matches(movie_name, movies_data['title'])
    if not find_close_match:
        return []
    close_match = find_close_match[0]
    index_of_the_movie = movies_data[movies_data['title'] == close_match]['index'].values[0]
    similarity_score = list(enumerate(similarity[index_of_the_movie]))
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)
    recommended_movies = []
    for movie in sorted_similar_movies:
        index = movie[0]
        title_from_index = movies_data[movies_data['index'] == index]['title'].values[0]
        recommended_movies.append(title_from_index)
    return recommended_movies

def get_related_videos(movie_list):
    video_data = []
    for movie in movie_list[:5]:  # Fetch videos for the first two movies only
        search_response = youtube.search().list(
            q=movie + ' movie',
            part='snippet',
            maxResults=1
        ).execute()

        for search_result in search_response.get('items', []):
            if search_result['id']['kind'] == 'youtube#video':
                video_data.append({
                    'title': movie,
                    'url': 'https://www.youtube.com/watch?v=' + search_result['id']['videoId'],
                    'thumbnail': search_result['snippet']['thumbnails']['medium']['url']
                })
                break  # Fetch only one video per movie

    return video_data

if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')
     