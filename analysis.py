import duckdb
import pandas as pd
lyrics = pd.read_parquet('data/song_lyrics_asian.parquet', engine="fastparquet")
conn = duckdb.connect(database=':memory:', read_only=False)
#Using asian songs data to extract top 100 songs from each region based on views
conn.register('lyrics_df', lyrics)
duckdb_result = conn.execute("""
    SELECT
    *
    FROM lyrics_df
    QUALIFY ROW_NUMBER() OVER (
    PARTITION BY region_group
    ORDER BY views DESC) <= 1000
    ORDER BY 
    region_group,
    views DESC

""").fetchdf()
conn.close()
print(f"Total tracks selected {len(duckdb_result)}")
print(duckdb_result.info())
output_file_path = 'data/duckdb_result.parquet'
duckdb_result.to_parquet(output_file_path, engine='fastparquet', index=False)

#spotify kaggle dataset and loading it into a parquet file
csv_file_path = 'data/train.csv'
train_df = pd.read_csv(csv_file_path)
lyrics_df = pd.read_parquet(output_file_path, engine="fastparquet")
print(f"Training DataFrame shape: {train_df.shape}")
lyrics_df['join_key'] = lyrics_df['title'].str.lower()
train_df['join_key'] = train_df['track_name'].str.lower()

conn = duckdb.connect(database=':memory:', read_only=False)
conn.register('lyrics_data', lyrics_df)
conn.register('train_data', train_df)


# joining them both together and now this is our final csv
joined_df = conn.execute("""
    SELECT
        t1.*,
        t2.popularity,
        t2.duration_ms,
        t2.acousticness,
        t2.explicit,
        t2.energy,
        t2.danceability,
        t2.key,
        t2.loudness,
        t2.mode,
        t2.speechiness,
        t2.instrumentalness,
        t2.liveness,
        t2.valence,
        t2.tempo
        -- Add other relevant columns from train_data here
    FROM
        lyrics_data AS t1
    JOIN
        train_data AS t2 ON t1.join_key = t2.join_key
""").fetchdf()

conn.close()

print("First 5 rows of the joined DataFrame:")
print(joined_df.head())
print(f"Joined DataFrame shape: {joined_df.shape}")
output_file_path = 'data/final.parquet'
joined_df.to_parquet(output_file_path, engine='fastparquet', index=False)
