import json


def build_genre_to_id(genres):
    genre_to_id = {"unknown": 0}
    for idx, genre in enumerate(genres, start=1):
        genre_to_id[genre] = idx
    return genre_to_id


def save_genre_to_id(genre_to_id, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(genre_to_id, f, indent=2, sort_keys=True)
