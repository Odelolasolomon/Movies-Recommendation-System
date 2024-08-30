# %% [markdown]
# # Collaborative Filtering Recommendation System

# %% [markdown]
# > In this project, you’ll use Jupyter Notebook, a web-based integrated development environment for Python as well as various Python libraries to create a recommendation system. The dataset is also available in the same directory as the Jupyter file. Every task in the project has one or more associated cells in the notebook

# %% [markdown]
# ## Task 1: Import Modules

# %% [markdown]
# - First, import all the libraries you’ll need for the project. You’ll need the following libraries:
# 
# - pandas: To store and manage data
# numpy: To handle all the numerical values in the dataset
# sklearn: To create the recommendation system
# cosine_similarity from sklearn.metrics.pairwise: To create a cosine similarity matrix

# %%
import pandas as pd 
import numpy as np 
import sklearn
from sklearn.metrics.pairwise import cosine_similarity

# %% [markdown]
# ## Task 2: Import the Dataset

# %% [markdown]
# - The dataset for this project is a [https://grouplens.org/datasets/movielens/100k], of which we are using two files: The dataset for this project is the MovieLens dataset. You’ll use the following two files from the dataset:
# 
# - For this task, import and load the dataset into a DataFrame as follows:
# 
# - Load Movie_data.csv and Movie_Id_Titles.csv in DataFrames.
# Join the DataFrames on Movie_ID.
# View the first five rows of the merged DataFrame.

# %%
#Load the rating data into a DataFrame:
column_names = ['User_ID', 'User_Names','Movie_ID','Rating','Timestamp']
movies_df = pd.read_csv('Movie_data.csv', sep = ',', names = column_names)

# %%
#Load the move information in a DataFrame:
movies_title_df = pd.read_csv("Movie_id_Titles-1.csv")
movies_title_df.rename(columns = {'item_id':'Movie_ID', 'title':'Movie_Title'}, inplace = True)

# %%
#Merge the DataFrames:
movies_df = pd.merge(movies_df,movies_title_df, on='Movie_ID')

#View the DataFrame:
print(movies_df.head())

# %% [markdown]
# ## Task 3: Explore the Dataset

# %% [markdown]
# - This project uses the dataset to create recommendations based on user similarity. The dataset should have sufficient ratings from the users to make good recommendations. For this task, explore the dataset and do the following:
# 
# - Get the dimensions of the DataFrame.
# - Get the statistical summary of the DataFrame.
# - Find the number of ratings given by each user.
# - Store the number of unique movies and users for the next task.

# %%
print(f"\n Size of the movie_df dataset is {movies_df.shape}")

# %%
movies_df.describe()

# %%
movies_df.groupby('User_ID')['Rating'].count().sort_values(ascending = True).head()

# %%
n_users = movies_df.User_ID.unique().shape[0]
n_movies = movies_df.Movie_ID.unique().shape[0]
print( str(n_users) + ' users')
print( str(n_movies) + ' movies')

# %% [markdown]
# ## Task 4: Create an Interaction Matrix

# %% [markdown]
# To create a collaborative filtering recommendation system, you need an interaction matrix to represent the relationship of every user with every movie in terms of ratings.
# 
# For this task, create a 2D array of nxm dimensions where n is the number of users and m is the number of movies. Next, place the ratings from DataFrame in the array.

# %%
#This would be a 2D array matrix to display user-movie_rating relationship
#Rows represent users by IDs, columns represent movies by IDs
ratings = np.zeros((n_users, n_movies))
for row in movies_df.itertuples():
    ratings[row[1], row[3]-1] = row[4]

# View the matrix
print(ratings)

# %% [markdown]
# ## Task 5: Explore the Interaction Matrix

# %% [markdown]
# One of the main characteristics of an interaction matrix is its density, which helps it provide good recommendations. The density of a dataset directly impacts the quality of the recommendations.
# 
# For this task, calculate the sparsity of the interaction matrix.

# %%
sparsity = float(len(ratings.nonzero()[0]))
sparsity /= (ratings.shape[0] * ratings.shape[1])
sparsity *= 100
print(sparsity)

# %% [markdown]
# ## Task 6 : Create a Similarity Matrix

# %% [markdown]
# User-user collaborative filtering is based on finding the similarity among users.
# 
# For this task, use cosine similarity to find the similarity among users.

# %%
rating_cosine_similarity = cosine_similarity(ratings)

# %% [markdown]
# ## Task 7: Provide Recommendations

# %% [markdown]
# Now that the cosine similarity matrix has been created, the system can recommend movies to the users according to their taste.
# 
# For this task, create a function that receives a user’s ID. Then, do the following to give movie recommendations to the user:
# 
# Find the k most similar users. Let’s assume k=10.
# Find the average rating of the movies rated by these k users.
# Find the top 10 rated movies.

# %%
def movie_recommender(user_item_m, X_user, user, k=10, top_n=10):
    # Get the location of the actual user in the User-Items matrix
    # Use it to index the User similarity matrix
    user_similarities = X_user[user]
    # obtain the indices of the top k most similar users
    most_similar_users = user_item_m.index[user_similarities.argpartition(-k)[-k:]]
    # Obtain the mean ratings of those users for all movies
    rec_movies = user_item_m.loc[most_similar_users].mean(0).sort_values(ascending=False)
    # Discard already seen movies
    m_seen_movies = user_item_m.loc[user].gt(0)
    seen_movies = m_seen_movies.index[m_seen_movies].tolist()
    rec_movies = rec_movies.drop(seen_movies).head(top_n)
    # return recommendations - top similar users rated movies
    rec_movies_a=rec_movies.index.to_frame().reset_index(drop=True)
    rec_movies_a.rename(columns={rec_movies_a.columns[0]: 'Movie_ID'}, inplace=True)
    return rec_movies_a

# %% [markdown]
# ## Task 8: View the Provided Recommendations 

# %% [markdown]
# For this task, run the function created in the previous task and view the recommendations provided to a user through the created system.
# 
# To complete this task, ensure the arguments provided to the function have the required data type to call the function

# %%
#Converting the 2D array into a DataFrame as expected by the movie_recommender function
ratings_df=pd.DataFrame(ratings)

# %%
user_ID=12
movie_recommender(ratings_df, rating_cosine_similarity,user_ID)

# %% [markdown]
# ## Task 9: Create Wrapper Function

# %% [markdown]
# This project aims to create an application that receives a User ID and provides all the recommendations for that specific user. For this, the recommendation function created in the Jupyter Notebook should be callable in a Python file via another function, i.e., a wrapper function.
# 
# For this task, perform the following operations:
# 
# Create another function, movie_recommender_run, that takes the user’s name and calls the recommendation function with the respective user ID.
# Use the output of the function call and return the list of recommendations in the form of Movie_ID and Movie_Title from movie_recommender_run.
# Save the notebook you’re working in to make it is usable in the next tasks.

# %%
def movie_recommender_run(user_Name):
    #Get ID from Name
    user_ID=movies_df.loc[movies_df['User_Names'] == user_Name].User_ID.values[0]
    #Call the function
    temp=movie_recommender(ratings_df, rating_cosine_similarity, user_ID)
    # Join with the movie_title_df to get the movie titles
    top_k_rec=temp.merge(movies_title_df, how='inner')
    return top_k_rec 


