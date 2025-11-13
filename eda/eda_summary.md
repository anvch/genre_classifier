# Exploratory Data Analysis 
Angela Chen, Samiksha Karimbil, Pragati Toppo 

---

## Dataset Selection 

We are selecting these datasets because they are pretty comprehensive and we don't have to do any web scraping to get this information. Especially for Spotify metadata, the features they have such as 'danceability' seem to be pretty interesting and could differentiate between different genres (since we generally hypothesize that genres such as C-pop are more emo and therefore less 'danceable'). However, we think other features for the lyrics also can differentiate genres (i.e. we have observed that K-pop has a lot of repetitive choruses). As such, our dataset will be made by joining the Genius song lyrics dataset with the Spotify Tracks Metadata dataset. However, this join narrows down the available data significantly as you will see below.

We are also really only interested in scoping to differentiate between Asian pop, and so we chose to only include songs that are Chinese/Japanese/Korean/Indian (which we filtered by language code).

Our final data we are analyzing can be found in ```data/asian_songs_translated_w_metadata_lyric_features.parquet```. The code to make this dataset can be found in ```eda/eda.ipynb```. To run the notebook, you must first ensure that you download the csv data from the links below and convert to parquet (respectively, name them ```song_lyrics.parquet``` and ```song_metadata.parquet```).

### Download Source Data:

Genius Song Lyrics: https://www.kaggle.com/datasets/carlosgdcj/genius-song-lyrics-with-language-information/data

Spotify Tracks Genre: https://www.kaggle.com/datasets/thedevastator/spotify-tracks-genre-dataset/data 

---

## Key variables, data volume, missingness, and potential target or interaction signals.

Key columns/variables:
title, tag (music type), artist, year, views (Genius lyrics views), popularity, duration_ms, acousticness, explicit, danceability, key, loudness, speechiness, liveness, valence, tempo, clean_lyrics, lyrics_translated, word_count, unique_words, repetition_ratio, lexical_diversity, sentiment_polarity, region_group

To keep the data balanced, we wanted to aim to have roughly the same number of songs within each genre. We were aiming for around 50 for each genre, but unfortunately Chinese is lacking with only 24 songs.

Song counts:   
Indian - 62   
Japanese - 50   
Korean - 50   
Chinese - 24   

For this data, however, at least all of the columns are filled.

---

## Informative visualizations

---

## Initial ideas for features and any anticipated challenges

The most promising features that we would like to explore out of all the available columns would be repetition_ratio, lexical_diversity, sentiment_polarity, tempo, key, duration_ms, and tag. We should avoid using the features of title, artist, clean_lyrics, and region_group because those give away the genre pretty explicitly rather than having us try to find the latent features underlying the different genres. region_group would be our ground truth as to what genre each song is. If the Spotify metadata seems to be not that promising or if we have too little data, we can attempt to just use the Genius song lyrics metadata with our extracted features because they have a lot more variety of songs. Another issue is that some of the songs we selected may be from a lesser variety of artists than we hope (i.e. a lot of the K-pop is BTS).