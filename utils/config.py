import json
import os


def load_config(path=None):
    if path is None:
        path = os.path.join("config", "config.json")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
