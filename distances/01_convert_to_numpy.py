"""
This script converts FMA music files to numpy arrays and saves them, so that distance code
doesn't have to.
"""


import os
import time

from pathlib import Path
from dotenv import (
    load_dotenv,
    find_dotenv
)

import pandas as pd
import numpy as np
import librosa
from fastdtw import fastdtw

def track_id_to_file_path(fma_small_dir, id, ext=".mp3") -> Path:
    str_id = str(id)
    num_digits = len(str_id)

    pad_size = 6 - num_digits

    filename = "0" * pad_size + str_id + ext

    return fma_small_dir/filename[0:3]/filename


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    fma_dir = Path(os.getenv("FMA_DIR"))
    fma_small_dir = fma_dir/"fma_small"
    fma_metadata_dir = fma_dir/"fma_metadata"
    fma_numpy_dir = fma_dir/"numpy"
    metadata = pd.read_csv(fma_metadata_dir/"tracks_small.csv", index_col="track_id")

    skips = []

    for track_id in metadata.index:

            try:
                print(f"Processing track id: {track_id}")
                track_start_time = time.time()
                track_filepath = track_id_to_file_path(fma_small_dir, track_id)

                # leaving out the sample rate parameter resamples the data
                # to 22050 hz
                track_data, track_sr = librosa.load(str(track_filepath))

                print(f"Metadata track length:{metadata.loc[track_id].track_duration}")
                print(f"Track sample rate: {track_sr}. Track numpy length: {track_data.shape}")

                np.save(fma_numpy_dir/track_filepath.name[0:6], track_data)
                track_end_time = time.time()

                print(f"Track processed in: {track_end_time-track_start_time}s")
            except ZeroDivisionError:
                skips.append(track_id)

    np.save(fma_numpy_dir/"skipped_ids.npy", np.array(skips))
