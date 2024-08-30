import pandas as pd
import numpy as np
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from ipynb.fs.full.CollaborativeFiltering import movie_recommender_run

#Set page configuration
st.set_page_config(layout = "wide", page_title = "Movie Recommendation App", page_icon = ":Cinema:")

#Write code to call movie_recommender_run and display recommendations
#Read the dataset to find unique users
column_names = ['User_ID', 'User_Names','Movie_ID','Rating','Timestamp']
movies_df = pd.read_csv('Movie_data.csv', sep = ',', names = column_names)
n_users = movies_df.User_Names.unique()

#Create application's header
st.header("Movies Recommendation System")

#Create a dropdown of UserIDs
User_Name = st.selectbox(
 "Select a user name:",
 (n_users)
)

st.write("This user might be interested in the following movies:")
#Find and display recommendations for selected users
result = movie_recommender_run(User_Name)
st.table(result.Movie_Title)
#Display movie rating charts here

# Display details of provided recommendations
ids= result.Movie_ID
Names=result.Movie_Title
fig = make_subplots(
    rows=5, cols=2,
    subplot_titles=(Names),
    specs=[
        [{"type": "bar"}, {"type": "bar"}],
        [{"type": "bar"}, {"type": "bar"}],
        [{"type": "bar"}, {"type": "bar"}],
        [{"type": "bar"}, {"type": "bar"}],
        [{"type": "bar"}, {"type": "bar"}] 
    ])

# x_row and y_col will determine the location of a plot in the plot-grid
x_row=1
y_col=1
for i in range (len(result)):
    temp=(movies_df.loc[movies_df['Movie_ID'] == ids[i]]).groupby('Rating').User_ID.count().reset_index()
    
    Rating=temp.Rating.to_numpy()
    User_ID= temp.User_ID.to_numpy()

    x_row= int( i/2 +1)
    y_col= i%2 + 1
    
    fig.add_trace(go.Bar(x=[1,2,3,4,5], y=User_ID), row=x_row, col=y_col)

fig.update_layout(height=900,width=800, showlegend=False, title= "Ratings of Suggested Movies")

st.plotly_chart(fig, use_container_width=True)
