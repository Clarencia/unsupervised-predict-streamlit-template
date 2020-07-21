"""

    Collaborative-based filtering for item recommendation.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: You are required to extend this baseline algorithm to enable more
    efficient and accurate computation of recommendations.

    !! You must not change the name and signature (arguments) of the
    prediction function, `collab_model` !!

    You must however change its contents (i.e. add your own collaborative
    filtering algorithm), as well as altering/adding any other functions
    as part of your improvement.

    ---------------------------------------------------------------------

    Description: Provided within this file is a baseline collaborative
    filtering algorithm for rating predictions on Movie data.

"""

# Script dependencies
import pandas as pd
import numpy as np
import pickle
import copy
from surprise import Reader, Dataset
from surprise import SVD, NormalPredictor, BaselineOnly, KNNBasic, NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Importing data
movies_df = pd.read_csv('resources/data/movies.csv',sep = ',',delimiter=',')
ratings = pd.read_csv('resources/data/ratings.csv')
ratings_df.drop(['timestamp'], axis=1,inplace=True)

# We make use of an SVD model trained on a subset of the MovieLens 10k dataset.
model=pickle.load(open('resources/models/SVD.pkl', 'rb'))

def prediction_item(item_id):
    """Map a given favourite movie to users within the
       MovieLens dataset with the same preference.

    Parameters
    ----------
    item_id : int
        A MovieLens Movie ID.

    Returns
    -------
    list
        User IDs of users with similar high ratings for the given movie.

    """
rate= pd.merge(ratings[['userId','movieId','rating']],movies_df[['title',"movieId"]],on = "movieId")
util_matrix = rate.pivot_table(index=['title'], columns=['userId'],values='rating')  

# Normalize each row (a given user's ratings) of the utility matrix
util_matrix_norm = util_matrix.apply(lambda x: (x-np.mean(x))/(np.max(x)-np.min(x)), axis=1)
# Fill Nan values with 0's, transpose matrix, and drop users with no ratings
util_matrix_norm.fillna(0, inplace=True)
util_matrix_norm = util_matrix_norm.T
util_matrix_norm = util_matrix_norm.loc[:, (util_matrix_norm != 0).any(axis=0)]
# Save the utility matrix in scipy's sparse matrix format
util_matrix_sparse = sp.sparse.csr_matrix(util_matrix_norm.values)

# Compute the similarity matrix using the cosine similarity metric
movie_similarity = cosine_similarity(util_matrix_sparse.T)
# Save the matrix as a dataframe to allow for easier indexing  
movie_sim_df = pd.DataFrame(movie_similarity,index = util_matrix_norm.columns,columns = util_matrix_norm.columns)
    
                           

def collab_model(movie_list,top_n=10):
    
    
    #select movie 1
    if movie_list[1] and movie_list[1] and movie_list[1] not in movie_sim_df.columns:
        recommended_movies=rate.groupby('title').mean().sort_values(by='rating', ascending=False).index[:N].to_list()

    else:
        movie1 = pd.DataFrame(movie_sim_df[movie_list[0]])
        movie1= movie1.reset_index()
        movie1['similarity']= movie1[movie_list[0]]
        movie1=pd.DataFrame(movie1,columns=['title','similarity'])


        #select movie 2
        movie2 = pd.DataFrame(movie_sim_df[movie_list[1]]) 
        movie2= movie2.reset_index()
        movie2['similarity']= movie2[movie_list[1]]
        movie2=pd.DataFrame(movie2,columns=['title','similarity'])


        #select movie 3
        movie3 = pd.DataFrame(movie_sim_df[movie_list[2]])
        movie3= movie3.reset_index()
        movie3['similarity']= movie3[movie_list[2]]
        movie3=pd.DataFrame(movie3,columns=['title','similarity'])


        finalmovies= pd.concat([movie1,movie2,movie3])
        recommended_movies=finalmovies.sort_values('similarity',ascending=False)
        recommended_movies=list(recommended_movies[3:13]['title'])
    return recommended_movies
