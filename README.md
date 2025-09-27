## Movie Recommendation System

This project is a simple **content-based movie recommender system** built using Python, Pandas, Scikit-learn, and TF-IDF vectorization.  
It suggests movies similar to a user’s input based on features like **genres, keywords, cast, director, tagline, and language**.

---

## Features
- Handles **typos in movie names** using `difflib`.
- Uses **TF-IDF Vectorizer** to convert text data into numerical vectors.
- Computes similarity between movies using **Cosine Similarity**.
- Returns the **Top 30 recommended movies**.
- Implemented in a clean, beginner-friendly way.

---

## Tech Stack
- Python 3
- Pandas, NumPy
- scikit-learn (TfidfVectorizer, cosine_similarity)
- difflib (for fuzzy title matching)
- Jupyter Notebook (for step-by-step explanation)

---


##  How It Works
1. Load the dataset (`movies.csv`).
2. Select important features: genres, keywords, tagline, cast, director, language.
3. Handle missing values by replacing NaN with empty strings.
4. Combine selected features into a single text column.
5. Convert text to vectors using **TF-IDF**.
6. Compute **cosine similarity** across all movies.
7. Take user input (movie name), find the closest match (using `difflib`).
8. Recommend the top 30 most similar movies.

---


## Dataset
The dataset used is **[TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)** (from Kaggle).  
It contains information about movies, including genres, keywords, cast, director, and more.


