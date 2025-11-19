import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

path = "./data/asian_songs_translated_w_metadata_lyric_features.parquet"
df = pd.read_parquet(path)
drop_cols = ["title", "artist", "year", "views", "features", "lyrics", "id", "language_cld3", "language_ft", "language", "join_key", "popularity", "explicit", "clean_lyrics"]

def load_and_prepare_data():
    # define label
    y = df["region_group"]
    # drop the label from features
    X = df.drop(columns=["region_group"])

    # drop unnecessary features
    for col in drop_cols:
        if col in X.columns:
            X = X.drop(columns=[col])

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test

def load_full_data():
    y = df["region_group"]
    X = df.drop(columns=["region_group"])
    X = X.drop(columns=[c for c in drop_cols if c in X.columns])
    return X, y