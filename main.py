import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
from functools import lru_cache
import ast
import random
import math
import os

from candidate import CandidateModel
import knn

"""
Data format:
creation-timestamp, latitude, longitude, likes, shares, comments, friends
"""


@lru_cache()
def location_to_coords(location: str, geolocator: Nominatim) -> tuple[int, int]:
    # calling the Nominatim tool and create Nominatim class

    # entering the location name
    getLoc = geolocator.geocode(location)
    if getLoc == None:
        return 0, 0

    return getLoc.latitude, getLoc.longitude


def update_csv(path: str) -> np.ndarray:
    loc = Nominatim(user_agent="Geopy Library", timeout=10)
    df = pd.read_csv(path)
    new = []
    for idx, row in df.iterrows():
        if pd.isna(df.at[idx, "likes"]) or pd.isna(df.at[idx, "comments"]):
            continue

        isFriends = 1 if random.randrange(1, 5) == 1 else 0
        shares = math.floor(int(row["likes"]) * (math.floor(random.random() + 0.75)))
        lat = 0.0
        lon = 0.0

        if not pd.isna(df.at[idx, "location"]):
            locData = ast.literal_eval(df.at[idx, "location"])
            lat, lon = location_to_coords(locData["name"], loc)

        new.append(
            [
                0.0 if pd.isna(value := df.at[idx, "created_at"]) else value,
                lat,
                lon,
                float(row["likes"]),
                shares,
                float(row["comments"]),
                isFriends,
            ]
        )
        print(idx) if idx % 100 == 0 else None
    np.savetxt("./dataset/post_data.csv", np.array(new, dtype=float), fmt="%.2f")


def load_csv(path: str) -> np.ndarray:
    df = pd.read_csv(path)
    print(f"Loaded csv of shape {df.shape}")
    npa = df.to_numpy(dtype=object)
    return npa


def reduce_dataset(path: str, n: int = 11500):
    """Reduces the csv file input to n rows

    Args:
        path: path to the file to reduce
        n: number of rows to reduce to
    """
    filename, ext = os.path.splitext(os.path.basename(path))
    filename = f"{filename}-reduced-{n}"

    df = pd.read_csv(path)
    df.fillna("", inplace=True)
    new_df = df.head(n)
    new_df.to_csv(
        f"{os.path.dirname(path)}/{filename}{ext}",
        quoting=1,
        quotechar='"',
        index=False,
    )


if __name__ == "__main__":
    training = load_csv("./dataset/post_data.csv")

    # model = CandidateModel(knn.Model("./models/null.pkl"), np.zeros(1))
    reduced_training = CandidateModel.reduce(training, language="en")
    print(reduced_training.shape)
