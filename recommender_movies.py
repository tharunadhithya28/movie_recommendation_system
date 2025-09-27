# %% [markdown]
# Content Based Recommendation System 
# 
# 1. Import the Necessary 
# 2. Loading the data and seeing the first five rows using head()
# 3. Selecting only the relevant features for recommendation 
# 4. Replacing the null values with null string(" ")
# 5. Combining all the selected features 
# 6. Now using TfidVectorizer -> convert all the text data to numeric 
# 7. Now using cosine_similarity, get the similarity scores of feature vectors
# 8. Getting the movie name as input from user 
# 9. Creating a list with all the movie names in the title 
# 10. Finding the close match with the title we got from input 
# 11. From the suggested close matches just take the first one alone 
# 12. Find the index of the final close match using the title 
# 13. With the index, find the list of similar movies using similarity (similarityscore we find)
# 14. Sorting the movies based on the similarity score(descending)
# 15. Print the movies based on the index 
# 

# %%
import numpy as np 
import pandas as pd 
import difflib 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity

# %%
movies_data = pd.read_csv("movies.csv")
movies_data.head()

# %%
movies_data.info()

# %%
selected_features = ['genres', 'keywords','original_language','tagline','cast','director']
print(selected_features)

# %%
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna(" ")

# %%
combined_features = movies_data['genres'] + movies_data['keywords'] + movies_data['tagline'] + movies_data['cast'] + movies_data['director']
print(combined_features)

# %%
vectorized = TfidfVectorizer()
feature_vectors = vectorized.fit_transform(combined_features)
print(feature_vectors)

# %%
similarity = cosine_similarity(feature_vectors)

print(similarity)

# %%
movie_name = input("Enter Movie Name : ")
print(movie_name)

# %%
list_of_all_titles = movies_data['title']
print(list_of_all_titles)

# %%
close_match = difflib.get_close_matches(movie_name, list_of_all_titles)[0]
print(close_match)

# %%
index_of_the_movie = movies_data[close_match == movies_data.title]['index'].values[0]
print(index_of_the_movie)

# %%
similar_movies = list(enumerate(similarity[index_of_the_movie]))
print(similar_movies)


# %%
sorted_similar_movies = sorted(similar_movies, key=lambda x:x[1], reverse=True)
print(sorted_similar_movies)

# %%
print("Movies Suggested for you : ")

i = 1 

for movie in sorted_similar_movies : 
    index = movie[0]
    title_from_index = movies_data[movies_data.index == index]['title'].values[0]
    if(i < 30):
        print(i,".",title_from_index)
        i+=1
    


