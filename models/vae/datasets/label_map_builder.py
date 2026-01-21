import glob
import json
import os

from models.vae.datasets.aist_filename_parser import get_genre_code


def build_genre_to_id(aist_dir, pattern="*.pkl"):
    files = sorted(glob.glob(os.path.join(aist_dir, pattern)))
    if not files:
        raise FileNotFoundError(f"No AIST++ files found in {aist_dir}")

    genres = sorted({get_genre_code(p) for p in files})
    genre_to_id = {"unknown": 0}
    for idx, genre in enumerate(genres, start=1):
        genre_to_id[genre] = idx
    return genre_to_id


def save_genre_to_id(genre_to_id, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(genre_to_id, f, indent=2, sort_keys=True)
