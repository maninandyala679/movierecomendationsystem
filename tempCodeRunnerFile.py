from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the movie dataset
movies = pd.read_csv('./movie_dataset.csv')

# Add 'tags' column combining genre and overview
movies['tags'] = movies['genre'] + movies['overview']

# Create a new dataframe with relevant columns
new_df = movies[['id', 'title', 'genre', 'overview', 'tags']]

# Initialize the CountVectorizer
cv = CountVectorizer(max_features=10000, stop_words='english')

# Create the vectors
vec = cv.fit_transform(new_df['tags'].values.astype('U')).toarray()

# Calculate the similarity matrix
sim = cosine_similarity(vec)

# Function to get movie recommendations
def recommend(movie):
    index = new_df[new_df['title'] == movie].index[0]
    distance = sorted(list(enumerate(sim[index])), reverse=True, key=lambda vec: vec[1])
    recommended_movies = []
    for i in distance[1:6]:
        recommended_movies.append(new_df.iloc[i[0]].title)
    return recommended_movies

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend_movie():
    movie = request.form['movie']
    recommendations = recommend(movie)
    return render_template('recommendations.html', movie=movie, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)

