import numpy as np
from pathlib import Path

def track_id_to_file_path(fma_base_dir, id, ext=".mp3") -> Path:
    str_id = str(id)
    num_digits = len(str_id)

    pad_size = 6 - num_digits

    filename = "0" * pad_size + str_id + ext

    return Path(fma_base_dir, filename[0:3], filename)