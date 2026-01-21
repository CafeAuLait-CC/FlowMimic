import os


def parse_aist_filename(filename):
    base = os.path.basename(filename)
    name = os.path.splitext(base)[0]
    parts = name.split("_")
    if not parts or not parts[0].startswith("g"):
        raise ValueError(f"Unexpected AIST filename: {filename}")
    genre_code = parts[0][1:]
    return {"genre_code": genre_code}


def get_genre_code(filename):
    return parse_aist_filename(filename)["genre_code"]
