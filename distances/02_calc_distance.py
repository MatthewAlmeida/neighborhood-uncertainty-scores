"""
This script calculates the dtw distances between songs and saves down the results.
"""

from pathlib import Path
import numpy as np
import argparse
import pandas as pd

from fastdtw import fastdtw

def track_id_to_file_path(fma_numpy_dir, id, ext=".mp3") -> Path:
    str_id = str(id)
    num_digits = len(str_id)

    pad_size = 6 - num_digits

    filename = "0" * pad_size + str_id + ext

    return fma_numpy_dir/filename

parser = argparse.ArgumentParser(description="Calculate and save sets of distances.")
parser.add_argument(
    "--start_idx", 
    "-si",
    type=int, 
    default=0,
    help="The index in the unrolled lower triangular matrix to begin calculating distances."
)
parser.add_argument(
    "--end_idx",
    "-ei",
    type=int,
    default=100,
    help="The index in the unrolled lower triangular matrix to stop calculating"
)
parser.add_argument(
    "--metadata_filename",
    "-mf",
    type=str,
    default="/data/FMA/fma_metadata/tracks_small.csv"
)
parser.add_argument(
    "--fma_small_dir",
    "-fmas",
    type=str,
    default="/data/FMA/fma_small/"
)
parser.add_argument(
    "--fma_numpy_dir",
    "-fman",
    type=str,
    default="/data/FMA/numpy/"
)
parser.add_argument(
    "--output_dir",
    "-od",
    type=str,
    default="/data/FMA/distances/"
)
parser.add_argument(
    "--worker_name",
    "-wn",
    type=str,
    default="worker"
)

if __name__ == "__main__":

    args = parser.parse_args()

    start_idx = args.start_idx
    end_idx = args.end_idx
    meta_fn = args.metadata_filename

    meta = pd.read_csv(meta_fn, index_col="track_id")

    file_a_indices, file_b_indices = np.tril_indices(8000, -1)

    # i indexes the entries in a lower triangular
    # matrix. The indices of that entry give the file
    # indices that should be compared for distance 
    # calculation. We use the metadata to map these 
    # indices to track ids.

    column_headers = [
        "tril_index_a",
        "tril_index_b",
        "track_index_a",
        "track_index_b",
        "dtw_distance"
    ]

    skip_headers = [
        "tril_index_a",
        "tril_index_b",
        "track_index_a",
        "track_index_b",
        "exception"
    ]

    results = []
    skips = []

    for i in range(start_idx, end_idx):
        if (i % 1000) == 0:
            print(f"{args.worker_name} processing index: {i}")

        # print("Indexing:")

        file_a_idx = file_a_indices[i]
        file_b_idx = file_b_indices[i]

        track_id_a = meta.index[file_a_idx]
        track_id_b = meta.index[file_b_idx]

        # print(f"Track a: Triangle index {file_a_idx}, track id {track_id_a}")
        # print(f"Track b: Triangle index {file_b_idx}, track id {track_id_b}")


        track_a_filename = track_id_to_file_path(
            Path(args.fma_numpy_dir), 
            track_id_a,
            ext=".npy"
        )

        track_b_filename = track_id_to_file_path(
            Path(args.fma_numpy_dir), 
            track_id_b,
            ext=".npy"
        )

        try:

            # print(f"Loading data files:")
            # print(f"  File a: {track_a_filename}")
            # print(f"  File b: {track_b_filename}")

            track_a_data = np.load(track_a_filename)
            track_b_data = np.load(track_b_filename)

            distance, _ = fastdtw(track_a_data, track_b_data)

            results.append(
                [
                    file_a_idx,
                    file_b_idx,
                    track_id_a,
                    track_id_b,
                    distance
                ]
            )

        except Exception as ex:
            skips.append(
                [
                    file_a_idx,
                    file_b_idx,
                    track_id_a,
                    track_id_b,
                    type(ex).__name__
                ]
            )

    results_df = pd.DataFrame(results, columns=column_headers)
    skips_df = pd.DataFrame(skips, columns=skip_headers)

    results_df.to_csv(
        args.output_dir + f"results_{args.start_idx}_{args.end_idx}.csv",
        index=False
    )

    skips_df.to_csv(
        args.output_dir + f"skips_{args.start_idx}_{args.end_idx}.csv",
        index=False
    )
