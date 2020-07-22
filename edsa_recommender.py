"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st

# Data handling dependencies
import pandas as pd
import numpy as np

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model

# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')

# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Welcome","About the App", "Recommender System","Solution Overview", "Data Visualisation","Contact Us" ]

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "Recommender System":
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('Fisrt Option',title_list[14930:15200])
        movie_2 = st.selectbox('Second Option',title_list[25055:25255])
        movie_3 = st.selectbox('Third Option',title_list[21100:21200])
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                           top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    if page_selection == "About the App":
        st.title("About the App")
        st.markdown("Are you a movie lover? Are you tired of wasting your time watching tons of trailers and ending up not watching their movies? Are you tired of finishing your popcorns before you find the right movie? Not anymore!!")
        st.image(["images/tired1.jpg", "images/tired22.jpg"],width=200)
        st.markdown("You have come to the right app.")
        #st.title("How to use the app")
        st.markdown(open('resources/data/information.md').read())
        
        
        
    if page_selection == "Solution Overview":
        st.title("Solution Overview")
        st.markdown("The app uses recommender systems to produce your recommendation in any of the two ways –")
        st.markdown("**Collaborative filtering**: which builds a model from your past behavior (i.e. movies watched or selected by the you) as well as similar decisions made by other users.")
        st.markdown("**Content-based filtering**: which uses a series of discrete characteristics of your selected movies in order to recommend additional movies with similar properties.")
        st.markdown("Below we have a video explaining more about recommender systems, how they work and why are recommender systems important.")
        if st.checkbox('View video'): # data is hidden if box is unchecked
            st.video('https://www.youtube.com/watch?v=U-yq3I9QugQ')
        
    if page_selection == "Welcome":
        st.markdown("![Alt Text](https://github.com/Clarencia/unsupervised-predict-streamlit-template/blob/master/images/welcome.gif?raw=true)")
        st.markdown("![Alt Text](https://cdn.clipart.email/11d8b5822102da1a7c7a2d015a569485_animated-popcorn-clipart-gif_350-350.gif)")
        
    if page_selection == "Contact Us":
        st.title("Connect with us")
        st.markdown('''<span style="color:green"> **Please help improve the app by rating it and telling us what you think could be changed to make your experience better.** </span>''', unsafe_allow_html=True)
        @st.cache(allow_output_mutation=True)
        def get_data():
            return []
        name = st.text_input("User name")
        inputs = st.text_input("How can we make your experience better?")
        rate = st.slider("Rate us", 0, 5)
        if st.button("Submit"):
            get_data().append({"User name": name, "Suggestion": inputs,"rating":rate})
        st.markdown('''<span style="color:green"> **What other users said:** </span>''', unsafe_allow_html=True)
        st.write(pd.DataFrame(get_data()))
        st.markdown('''<span style="color:green"> **For any questions contact us here:** </span>''', unsafe_allow_html=True)
        st.image(["images/contact.PNG"],width=800)
        
        
    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.


if __name__ == '__main__':
    main()
