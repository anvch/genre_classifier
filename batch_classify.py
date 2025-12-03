#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from data_utils import load_full_data, drop_cols


# Path to ENGLISH songs file
script_dir = os.path.dirname(os.path.abspath(__file__))
english_path = os.path.join(
    script_dir,
    "data",
    "final_joined_english.parquet"   # change if your english file has another name
)


def main():
    
    print("Loading Asian training data...")
    X_asian, y_asian = load_full_data()   # X_asian, y_asian from asian_songs_*.parquet
    print("Asian data shape:", X_asian.shape)
    print("Example labels:", y_asian.unique())

 
    X_asian_num = X_asian.select_dtypes(include=[np.number])

    print("\nLoading English songs...")
    eng_df = pd.read_parquet(english_path)
    print("English data shape:", eng_df.shape)

    # Start from a copy
    X_eng = eng_df.copy()

    # Drop any label column if present
    if "region_group" in X_eng.columns:
        X_eng = X_eng.drop(columns=["region_group"])

    # Drop the same non-feature columns as in data_utils
    for col in drop_cols:
        if col in X_eng.columns:
            X_eng = X_eng.drop(columns=[col])

    # Keep only numeric for English too
    X_eng_num = X_eng.select_dtypes(include=[np.number])

    
    common_cols = sorted(set(X_asian_num.columns) & set(X_eng_num.columns))
    if not common_cols:
        raise RuntimeError("No common numeric feature columns between Asian and English data!")

    print("\nUsing common numeric feature columns:")
    print(common_cols)

    X_asian_final = X_asian_num[common_cols]
    X_eng_final = X_eng_num[common_cols]

    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_asian)
    class_names = list(le.classes_)
    print("\nClasses:", class_names)

 
    print("\nTraining RandomForestClassifier on Asian songs...")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        class_weight="balanced"
    )
    rf.fit(X_asian_final, y_encoded)


    n_sample = min(50, len(eng_df))
    sample_df = eng_df.sample(n_sample, random_state=42)
    sample_idx = sample_df.index
    X_sample = X_eng_final.loc[sample_idx]

   
    print(f"\nClassifying {n_sample} English songs...\n")
    proba_all = rf.predict_proba(X_sample)   # shape: (n_sample, n_classes)

    for i, idx in enumerate(sample_idx):
        row_meta = sample_df.loc[idx]
        title = row_meta.get("title", "<unknown>")
        artist = row_meta.get("artist", "<unknown>")

        print(f" {title} — {artist}")
        song_proba = proba_all[i]

        # sort most-likely → least-likely
        order = np.argsort(song_proba)[::-1]

        for j in order:
            region = class_names[j]
            p = song_proba[j] * 100
            print(f"   {region:8s}: {p:5.2f}%")
        print()

    print("Done ")


if __name__ == "__main__":
    main()
