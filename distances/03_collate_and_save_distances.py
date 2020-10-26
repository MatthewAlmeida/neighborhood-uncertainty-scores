"""
This script collates the computed distances saved down by the 02 script and saves the 
distance matrix.
"""

import argparse
import os
import numpy as np
import pandas as pd
from pathlib import Path

_DATA_DIR = Path("/data/FMA/distances")

parser = argparse.ArgumentParser(description="Calculate and save sets of distances.")
parser.add_argument(
    "--save", 
    "-s",
    action="store_true"
)
if __name__ == "__main__":
    args = parser.parse_args()

    filenames = [fn for fn in os.listdir(_DATA_DIR) if fn.startswith("results")]

    distance_matrix = np.full((8000,8000), np.nan, dtype=np.float32)

    for filename in filenames:
        print(f"Loading distances from {_DATA_DIR/filename}")
        distances = pd.read_csv(_DATA_DIR/filename)

        distance_matrix[distances.tril_index_a, distances.tril_index_b] = distances.dtw_distance
        distance_matrix[distances.tril_index_b, distances.tril_index_a] = distances.dtw_distance

    if args.save:
        print("Saving matrix...")
        np.save(_DATA_DIR/"distance_matrix.npy", distance_matrix)

