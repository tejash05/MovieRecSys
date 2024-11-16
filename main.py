import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import urllib.request
import pickle
import certifi
import os
import ssl
os.environ['SSL_CERT_FILE'] = certifi.where()
app = Flask(__name__)

# Load the trained TF-IDF vectorizer and classifier
try:
    with open('tranform.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('nlp_model.pkl', 'rb') as f:
        clf = pickle.load(f)
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    vectorizer = None
    clf = None

def create_similarity():
    data = pd.read_csv('main_data.csv')
    # Creating a count matrix
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data['comb'])
    # Creating a similarity score matrix
    similarity = cosine_similarity(count_matrix)
    return data, similarity

def rcmd(m, data=None, similarity=None):
    m = m.lower()
    if data is None or similarity is None:
        data, similarity = create_similarity()
    if m not in data['movie_title'].str.lower().unique():
        return 'Sorry! Try another movie name.'
    else:
        i = data.loc[data['movie_title'].str.lower() == m].index[0]
        lst = list(enumerate(similarity[i]))
        lst = sorted(lst, key=lambda x: x[1], reverse=True)
        lst = lst[1:11]  # Exclude the first one as it's the requested movie
        recommended_movies = [data['movie_title'][i[0]] for i in lst]
        return recommended_movies

def convert_to_list(my_list):
    my_list = my_list.strip('["').strip('"]').split('","')
    return my_list

def get_suggestions():
    data = pd.read_csv('main_data.csv')
    return list(data['movie_title'].str.capitalize())

@app.route("/")
@app.route("/home")
def home():
    suggestions = get_suggestions()
    return render_template('home.html', suggestions=suggestions)

@app.route("/similarity", methods=["POST"])
def similarity_route():
    movie = request.form['name']
    rc = rcmd(movie)
    if isinstance(rc, str):
        return rc
    else:
        m_str = "---".join(rc)
        return m_str

@app.route("/recommend", methods=["POST"])
@app.route("/recommend", methods=["POST"])
def recommend():
    # Getting data from AJAX request
    title = request.form['title']
    cast_ids = request.form['cast_ids']
    cast_names = request.form['cast_names']
    cast_chars = request.form['cast_chars']
    cast_bdays = request.form['cast_bdays']
    cast_bios = request.form['cast_bios']
    cast_places = request.form['cast_places']
    cast_profiles = request.form['cast_profiles']
    imdb_id = request.form['imdb_id']
    poster = request.form['poster']
    genres = request.form['genres']
    overview = request.form['overview']
    vote_average = request.form['rating']
    vote_count = request.form['vote_count']
    release_date = request.form['release_date']
    runtime = request.form['runtime']
    status = request.form['status']
    rec_movies = request.form['rec_movies']
    rec_posters = request.form['rec_posters']

    # Get movie suggestions for autocomplete
    suggestions = get_suggestions()

    # Convert string representations to lists
    rec_movies = convert_to_list(rec_movies)
    rec_posters = convert_to_list(rec_posters)
    cast_names = convert_to_list(cast_names)
    cast_chars = convert_to_list(cast_chars)
    cast_profiles = convert_to_list(cast_profiles)
    cast_bdays = convert_to_list(cast_bdays)
    cast_bios = convert_to_list(cast_bios)
    cast_places = convert_to_list(cast_places)

    # Ensure all lists have the same length by trimming to the shortest list length
    min_length = min(len(cast_names), len(cast_ids), len(cast_profiles), len(cast_bdays), len(cast_places),
                     len(cast_bios))

    cast_ids = cast_ids[:min_length]
    cast_names = cast_names[:min_length]
    cast_chars = cast_chars[:min_length]
    cast_profiles = cast_profiles[:min_length]
    cast_bdays = cast_bdays[:min_length]
    cast_bios = cast_bios[:min_length]
    cast_places = cast_places[:min_length]

    # Combine multiple lists into dictionaries
    movie_cards = {rec_posters[i]: rec_movies[i] for i in range(len(rec_posters))}
    casts = {cast_names[i]: [cast_ids[i], cast_chars[i], cast_profiles[i]] for i in range(len(cast_profiles))}
    cast_details = {
        cast_names[i]: [cast_ids[i], cast_profiles[i], cast_bdays[i], cast_places[i], cast_bios[i]]
        for i in range(len(cast_places))
    }

    # Web scraping to get user reviews from IMDb site
    url = f'https://imdb-rest-api.herokuapp.com/api/livescraper/reviews/{imdb_id}'
    print(f"Fetching reviews from URL: {url}")

    # Fetch and parse JSON data from the API
    try:
        with urllib.request.urlopen(url) as response:
            sauce = response.read()
            print(f"Raw JSON response: {sauce}")
    except Exception as e:
        print(f"Error fetching data: {e}")
        sauce = b''  # Assign an empty byte string in case of error

    if sauce:
        try:
            data = json.loads(sauce)
            print("Successfully parsed JSON data.")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            data = {}
    else:
        data = {}

    # Extract reviews from the JSON data
    reviews_list = []
    reviews_status = []
    reviews = data.get('reviews', [])
    for review in reviews:
        review_text = review.get('full_review') or review.get('short_review')
        if review_text:
            reviews_list.append(review_text)
            # Prepare the review text for prediction
            movie_review_list = np.array([review_text])
            # Transform the review text using the vectorizer
            if vectorizer and clf:
                movie_vector = vectorizer.transform(movie_review_list)
                # Predict the sentiment using the classifier
                pred = clf.predict(movie_vector)
                # Append 'Good' or 'Bad' based on prediction
                reviews_status.append('Good' if pred[0] == 1 else 'Bad')
            else:
                reviews_status.append('Unknown')  # Handle cases where models aren't loaded

    print("Reviews List:", reviews_list)
    print("Reviews Status:", reviews_status)

    # Combine reviews and their statuses into a dictionary
    movie_reviews = {reviews_list[i]: reviews_status[i] for i in range(len(reviews_list))}

    # Passing all the data to the HTML file
    return render_template(
        'recommend.html',
        title=title,
        poster=poster,
        overview=overview,
        vote_average=vote_average,
        vote_count=vote_count,
        release_date=release_date,
        runtime=runtime,
        status=status,
        genres=genres,
        movie_cards=movie_cards,
        reviews=movie_reviews,
        casts=casts,
        cast_details=cast_details
    )


if __name__ == '__main__':
    app.run(debug=True)
