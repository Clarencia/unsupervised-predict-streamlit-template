"""

    Content-based filtering for item recommendation.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: You are required to extend this baseline algorithm to enable more
    efficient and accurate computation of recommendations.

    !! You must not change the name and signature (arguments) of the
    prediction function, `content_model` !!

    You must however change its contents (i.e. add your own content-based
    filtering algorithm), as well as altering/adding any other functions
    as part of your improvement.

    ---------------------------------------------------------------------

    Description: Provided within this file is a baseline content-based
    filtering algorithm for rating predictions on Movie data.

"""

# Script dependencies
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
# Importing data
movies = pd.read_csv('resources/data/movies.csv', sep = ',',delimiter=',')
movies_df=movies
movies_df = movies_df.drop(movies_df.loc[movies_df["title"].duplicated(keep='first') == True].index)

def data_preprocessing(subset_size):
    """Prepare data for use within Content filtering algorithm.

    Parameters
    ----------
    subset_size : int
        Number of movies to use within the algorithm.

    Returns
    -------
    Pandas Dataframe
        Subset of movies selected for content-based filtering.

    """
    # Split genre data into individual words.
    movies['keyWords'] = movies['genres'].str.replace('|', ' ')
    # Subset of the data
    movies_subset = movies[:subset_size]
    return movies_subset

# !! DO NOT CHANGE THIS FUNCTION SIGNATURE !!
# You are, however, encouraged to change its content.  
def content_model(movie_list,top_n=10):
    df=movies_df[:27000]
    df['genres'] = df['genres'].str.replace('|', ' ')
    tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
    tfidf_matrix = tf.fit_transform(df['genres'])
    tfidf_matrix.shape
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    df = df.reset_index()
    df=df.drop('index', axis=1)
    titles = df['title']
    indices = pd.Series(df.index, index=df['title'])

    #first movie
    idx1 = indices[movie_list[0]]
    sim_scores1 = list(enumerate(cosine_sim[idx1]))
    sim_scores1 = sorted(sim_scores1, key=lambda x: x[1], reverse=True)
    sim_scores1 = sim_scores1[0:1]

    #second movie
    idx2 = indices[movie_list[1]]
    sim_scores2 = list(enumerate(cosine_sim[idx2]))
    sim_scores2 = sorted(sim_scores2, key=lambda x: x[1], reverse=True)
    sim_scores2 = sim_scores2[0:1]

    #third movie
    idx3 = indices[movie_list[2]]
    sim_scores3 = list(enumerate(cosine_sim[idx3]))
    sim_scores3 = sorted(sim_scores3, key=lambda x: x[1], reverse=True)
    sim_scores3 = sim_scores3[0:1]
    
    mix= sim_scores3+sim_scores2+sim_scores1
    mix=sorted(mix,key=lambda x: x[1],reverse=True)
    movie_indices = [i[0] for i in mix]
    return [list(titles.iloc[movie_indices])[0]]