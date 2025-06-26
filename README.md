# Collaborative Filtering Recommendation System

![Python](https://img.shields.io/badge/python-3.7%2B-blue.svg)
![Jupyter](https://img.shields.io/badge/jupyter-notebook-orange.svg)

## Overview

This project implements a **User-User Collaborative Filtering Recommendation System** using the MovieLens 100K dataset. The system analyzes user preferences and movie ratings to provide personalized movie recommendations based on similarity between users. By identifying users with similar tastes, the system can suggest movies that a user is likely to enjoy based on what similar users have rated highly.

## 🎯 Objectives

- Build a collaborative filtering recommendation engine from scratch
- Implement user-user similarity using cosine similarity
- Create an interactive movie recommendation system
- Analyze dataset sparsity and its impact on recommendation quality
- Provide both user ID and user name-based recommendation interfaces

## 🌟 Key Features

### Core Functionality
- **User-User Collaborative Filtering**: Recommends movies based on similar user preferences
- **Cosine Similarity**: Uses cosine similarity to measure user similarity
- **Flexible Recommendations**: Configurable number of similar users (k) and recommendations (top_n)
- **Dual Interface**: Support for both User ID and User Name-based queries
- **Movie Filtering**: Automatically excludes movies already rated by the user

### Technical Features
- **Sparse Matrix Handling**: Efficiently processes sparse user-item interaction matrices
- **Scalable Architecture**: Designed to handle large datasets with optimized NumPy operations
- **Data Integration**: Seamlessly merges rating and movie metadata
- **Statistical Analysis**: Provides comprehensive dataset exploration and sparsity analysis

### Dataset Files
1. **Movie_data.csv**: Contains user ratings with columns:
   - `User_ID`: Unique user identifier
   - `User_Names`: User names
   - `Movie_ID`: Unique movie identifier
   - `Rating`: User rating (1-5)
   - `Timestamp`: Rating timestamp

2. **Movie_id_Titles-1.csv**: Contains movie metadata:
   - `item_id`: Movie ID (renamed to Movie_ID)
   - `title`: Movie title (renamed to Movie_Title)

## 🛠️ Installation

### Prerequisites
- Python 3.7 or higher
- Jupyter Notebook or JupyterLab
- Required Python libraries (see requirements below)

### Quick Setup
```bash
# Clone the repository
git clone https://github.com/your-username/collaborative-filtering-recommender.git
cd collaborative-filtering-recommender

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install pandas numpy scikit-learn jupyter matplotlib seaborn

# Launch Jupyter Notebook
jupyter notebook
```

### Requirements
```txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
jupyter>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## 🚀 Usage

### 1. Basic Recommendation by User ID
```python
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load and process data (follow Tasks 1-6 in notebook)
# ...

# Get recommendations for User ID 12
user_ID = 12
recommendations = movie_recommender(ratings_df, rating_cosine_similarity, user_ID)
print(recommendations)
```

### 2. Recommendation by User Name
```python
# Get recommendations using user name
user_name = "Shawn Wilson"
recommendations = movie_recommender_run(user_name)
print(recommendations[['Movie_ID', 'Movie_Title']])
```

### 3. Custom Parameters
```python
# Customize recommendation parameters
recommendations = movie_recommender(
    user_item_m=ratings_df, 
    X_user=rating_cosine_similarity, 
    user=user_ID, 
    k=15,      # Use 15 most similar users
    top_n=20   # Return top 20 recommendations
)
```

### 4. Complete Workflow Example
```python
# Complete recommendation workflow
def get_movie_recommendations(user_name, k=10, top_n=10):
    """
    Get movie recommendations for a user by name
    
    Args:
        user_name (str): Name of the user
        k (int): Number of similar users to consider
        top_n (int): Number of recommendations to return
    
    Returns:
        DataFrame: Movie recommendations with ID and Title
    """
    # Get user ID from name
    user_id = movies_df.loc[movies_df['User_Names'] == user_name, 'User_ID'].values[0]
    
    # Get recommendations
    rec_ids = movie_recommender(ratings_df, rating_cosine_similarity, user_id, k, top_n)
    
    # Merge with movie titles
    recommendations = rec_ids.merge(movies_title_df, on='Movie_ID', how='inner')
    
    return recommendations[['Movie_ID', 'Movie_Title']]

# Example usage
recommendations = get_movie_recommendations("Shawn Wilson", k=15, top_n=10)
print(recommendations)
```

 advanced_examples.py
```

## 🔍 Step-by-Step Implementation

### Task 1: Environment Setup
```python
# Import required libraries
import pandas as pd 
import numpy as np 
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
```

### Task 2: Data Loading and Integration
```python
# Load rating data
column_names = ['User_ID', 'User_Names','Movie_ID','Rating','Timestamp']
movies_df = pd.read_csv('Movie_data.csv', sep=',', names=column_names)

# Load movie titles
movies_title_df = pd.read_csv("Movie_id_Titles-1.csv")
movies_title_df.rename(columns={'item_id':'Movie_ID', 'title':'Movie_Title'}, inplace=True)

# Merge datasets
movies_df = pd.merge(movies_df, movies_title_df, on='Movie_ID')
```

### Task 3: Dataset Exploration
```python
# Dataset dimensions
print(f"Dataset shape: {movies_df.shape}")

# Statistical summary
print(movies_df.describe())

# User activity analysis
user_activity = movies_df.groupby('User_ID')['Rating'].count().sort_values(ascending=True)
print(f"Most active user rated {user_activity.max()} movies")
print(f"Least active user rated {user_activity.min()} movies")

# Store unique counts
n_users = movies_df.User_ID.unique().shape[0]
n_movies = movies_df.Movie_ID.unique().shape[0]
```

### Task 4: Interaction Matrix Creation
```python
# Create user-item interaction matrix
ratings = np.zeros((n_users, n_movies))
for row in movies_df.itertuples():
    ratings[row[1], row[3]-1] = row[4]  # row[3]-1 because Movie_ID starts from 1
```

### Task 5: Sparsity Analysis
```python
# Calculate matrix sparsity
sparsity = float(len(ratings.nonzero()[0])) / (ratings.shape[0] * ratings.shape[1]) * 100
print(f"Matrix sparsity: {sparsity:.2f}%")
```

### Task 6: Similarity Matrix
```python
# Compute cosine similarity between users
rating_cosine_similarity = cosine_similarity(ratings)
```

### Tasks 7-9: Recommendation Functions
```python
def movie_recommender(user_item_m, X_user, user, k=10, top_n=10):
    """Core recommendation function"""
    user_similarities = X_user[user]
    most_similar_users = user_item_m.index[user_similarities.argpartition(-k)[-k:]]
    rec_movies = user_item_m.loc[most_similar_users].mean(0).sort_values(ascending=False)
    
    # Remove already seen movies
    m_seen_movies = user_item_m.loc[user].gt(0)
    seen_movies = m_seen_movies.index[m_seen_movies].tolist()
    rec_movies = rec_movies.drop(seen_movies).head(top_n)
    
    rec_movies_df = rec_movies.index.to_frame().reset_index(drop=True)
    rec_movies_df.rename(columns={rec_movies_df.columns[0]: 'Movie_ID'}, inplace=True)
    return rec_movies_df

def movie_recommender_run(user_name):
    """Wrapper function for name-based recommendations"""
    user_id = movies_df.loc[movies_df['User_Names'] == user_name, 'User_ID'].values[0]
    temp = movie_recommender(ratings_df, rating_cosine_similarity, user_id)
    top_k_rec = temp.merge(movies_title_df, how='inner')
    return top_k_rec
```

## 📈 Performance Analysis

### Dataset Statistics
- **Total Ratings**: 100,003
- **Average Rating**: 3.53/5.0
- **Rating Distribution**: Most ratings are 3 or 4 stars
- **User Engagement**: Ranges from 3 to 737 ratings per user
- **Matrix Density**: 6.30% (93.70% sparse)

### Algorithm Performance
- **Time Complexity**: O(n²) for similarity computation, O(n) for recommendations
- **Space Complexity**: O(n²) for similarity matrix storage
- **Scalability**: Suitable for datasets up to ~10K users

### Recommendation Quality Factors
1. **Sparsity Impact**: Lower sparsity generally improves recommendations
2. **k Parameter**: Higher k values smooth recommendations but may reduce personalization
3. **User Activity**: Users with more ratings get better recommendations
4. **Cold Start**: New users with few ratings receive less accurate recommendations

## 🧪 Testing and Validation

### Unit Tests
```python
# Test recommendation function
def test_movie_recommender():
    # Test with known user
    recommendations = movie_recommender(ratings_df, rating_cosine_similarity, 12)
    assert len(recommendations) <= 10
    assert 'Movie_ID' in recommendations.columns
    print("✓ Basic recommendation test passed")

# Test wrapper function
def test_movie_recommender_run():
    recommendations = movie_recommender_run("Shawn Wilson")
    assert len(recommendations) <= 10
    assert all(col in recommendations.columns for col in ['Movie_ID', 'Movie_Title'])
    print("✓ Name-based recommendation test passed")
```

### Manual Validation
```python
# Check if recommendations make sense for a specific user
user_name = "Shawn Wilson"
user_ratings = movies_df[movies_df['User_Names'] == user_name]
print(f"\n{user_name}'s top rated movies:")
print(user_ratings.nlargest(5, 'Rating')[['Movie_Title', 'Rating']])

recommendations = movie_recommender_run(user_name)
print(f"\nRecommendations for {user_name}:")
print(recommendations)
```

## 🔧 Advanced Usage

### Custom Similarity Metrics
```python
from sklearn.metrics.pairwise import pearson_correlation, manhattan_distances

# Alternative similarity measures
def create_similarity_matrix(ratings, method='cosine'):
    if method == 'cosine':
        return cosine_similarity(ratings)
    elif method == 'pearson':
        return np.corrcoef(ratings)
    elif method == 'manhattan':
        return 1 / (1 + manhattan_distances(ratings))
    else:
        raise ValueError("Unsupported similarity method")
```

### Recommendation Evaluation
```python
def evaluate_recommendations(test_users, k=10, top_n=10):
    """Evaluate recommendation quality using precision@k"""
    precisions = []
    
    for user_id in test_users:
        # Split user's ratings into train/test
        user_ratings = ratings_df.iloc[user_id]
        rated_movies = user_ratings[user_ratings > 0].index.tolist()
        
        if len(rated_movies) >= 10:  # Ensure sufficient ratings
            # Hide some ratings for testing
            test_movies = np.random.choice(rated_movies, 3, replace=False)
            
            # Get recommendations (excluding test movies)
            recommendations = movie_recommender(ratings_df, rating_cosine_similarity, user_id, k, top_n)
            
            # Calculate precision
            hits = len(set(recommendations['Movie_ID'].tolist()) & set(test_movies))
            precision = hits / len(test_movies)
            precisions.append(precision)
    
    return np.mean(precisions)
```

### Batch Recommendations
```python
def batch_recommendations(user_list, k=10, top_n=10):
    """Generate recommendations for multiple users"""
    results = {}
    
    for user_name in user_list:
        try:
            recommendations = movie_recommender_run(user_name)
            results[user_name] = recommendations
        except IndexError:
            results[user_name] = f"User '{user_name}' not found"
    
    return results

# Example usage
users = ["Shawn Wilson", "Robert Poulin", "Laura Krulik"]
batch_results = batch_recommendations(users)
```

## 🔄 System Improvements

### Potential Enhancements
1. **Hybrid Approaches**: Combine collaborative filtering with content-based filtering
2. **Matrix Factorization**: Implement SVD or NMF for better scalability
3. **Deep Learning**: Use neural collaborative filtering
4. **Real-time Updates**: Implement incremental learning for new ratings
5. **Popularity Bias**: Address popularity bias in recommendations

### Performance Optimizations
```python
# Sparse matrix implementation for memory efficiency
from scipy.sparse import csr_matrix

def create_sparse_matrix(movies_df, n_users, n_movies):
    """Create sparse user-item matrix"""
    rows, cols, data = [], [], []
    
    for _, row in movies_df.iterrows():
        rows.append(row['User_ID'])
        cols.append(row['Movie_ID'] - 1)  # Movie_ID starts from 1
        data.append(row['Rating'])
    
    return csr_matrix((data, (rows, cols)), shape=(n_users, n_movies))

## 🤝 Contributing

We welcome contributions to improve the recommendation system! Here's how you can help:

### Ways to Contribute
- **Bug Reports**: Report issues or unexpected behavior
- **Feature Requests**: Suggest new features or improvements
- **Code Contributions**: Submit pull requests with enhancements
- **Documentation**: Improve documentation and examples
- **Testing**: Add test cases and validation scenarios

### Development Setup
```bash
# Fork the repository and clone your fork
git clone https://github.com/your-username/collaborative-filtering-recommender.git

# Create a feature branch
git checkout -b feature-name

# Make changes and test
python -m pytest tests/

# Submit pull request
```


## 🙏 Acknowledgments

### Dataset
- **GroupLens Research**: For providing the MovieLens 100K dataset
- **University of Minnesota**: For supporting recommender systems research

### Libraries and Tools
- **pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **Jupyter**: Interactive development environment


---

**Built with ❤️ for the Recommender Systems Community**

*Helping users discover movies they'll love through the power of collaborative filtering.*
