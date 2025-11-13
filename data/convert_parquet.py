import polars as pl

csv_path = "./song_lyrics.csv"
parquet_path = "./song_lyrics.parquet"

pl.scan_csv(csv_path).sink_parquet(parquet_path)

