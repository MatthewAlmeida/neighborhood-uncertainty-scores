import numpy as np
import pandas as pd
import time
import warnings
from sklearn.preprocessing import (
    OneHotEncoder, StandardScaler, MinMaxScaler
)
from pathlib import Path
from score_calculation.sample_weights import get_weight_vector_from_scores
from score_calculation.border_scores import get_border_scores_precomputed_distance

# Change this folder to the top-level FMA folder!
_root_folder = "/tf/workspace/FMA/"
_feature_csv_path = Path(f"{_root_folder}fma_metadata/features_single_header.csv")
_genre_csv_path = Path(f"{_root_folder}fma_metadata/genres.csv")
_meta_filename = Path(f"{_root_folder}fma_metadata/tracks_small.csv")


def default_sample_weights_for_training(y_int, examples_per_song,
    k=10, alpha=1.0, beta=None, gamma=1.75, eta=0.25, 
    use_numerator_distance=False, use_denominator_distance= False, 
    normalize_distance=True
):
    training_set, _, _ = get_boolean_selection_vectors(1)

    D = np.load(_distance_matrix_filename)

    sample_weights = get_sample_weights(training_set, y_int, D,
        k, alpha, beta, gamma, eta, use_numerator_distance,
        use_denominator_distance, normalize_distance
    )

    return np.repeat(sample_weights, examples_per_song)

def get_sample_weights(boolean_set_indices, 
    y_set,
    D,
    k=10, alpha=1.0, 
    beta=None,
    gamma=1.75,
    eta=0.25,
    use_numerator_distance=False,
    use_denominator_distance= False, 
    normalize_distance=True):

    train_and_valid_border_scores = calc_scores(boolean_set_indices, 
                                        y_set,
                                        D, k, use_numerator_distance,
                                        use_denominator_distance,
                                        normalize_distance)    

    # if no value passed, beta is set to the median of nonzero scores
    if not beta:
        beta = np.median(train_and_valid_border_scores[train_and_valid_border_scores > 0])

    sample_weights = get_weight_vector_from_scores(train_and_valid_border_scores,
                                                        alpha, beta, gamma, eta)
    return sample_weights

def calc_scores(boolean_set_indices, y_set, D, k, 
                use_numerator_distance, use_denominator_distance,
               normalize_distance):

    # Convert boolean indexes to integer so we can slice the
    # right distances out of the distance array.

    integer_set_indices = np.nonzero(boolean_set_indices)[0]
    integer_set_indices_column = integer_set_indices[:, np.newaxis]
    
    D_set = D[integer_set_indices_column, integer_set_indices]

    border_scores = get_border_scores_precomputed_distance(
        y_set, D_set, k,
        use_numerator_distance=use_numerator_distance,
        use_denominator_distance=use_denominator_distance,
        normalize_distance=normalize_distance
    )

    return border_scores

def track_id_to_file_path(fma_base_dir, id, ext=".mp3") -> Path:
    str_id = str(id)
    num_digits = len(str_id)

    pad_size = 6 - num_digits

    filename = "0" * pad_size + str_id + ext

    return Path(fma_base_dir, filename[0:3], filename)

def get_boolean_selection_vectors(examples_per_song=19):
    meta = pd.read_csv(_meta_filename, index_col="track_id")

    # Recover recommended training / valid / test splits
    # from metadata file

    training_set = np.repeat(meta.set_split == "training", examples_per_song).values
    validation_set = np.repeat(meta.set_split == "validation", examples_per_song).values
    test_set = np.repeat(meta.set_split == "test", examples_per_song).values

    return training_set, validation_set, test_set

def get_densenet_samples(
    scale="standard", 
    remove_nan = True, 
    mode="valid",
    calc_scores = False,
    data="std",
    dist_matrix="orig"
):
    """
    Paths to important files.

    feature_csv: this file conatins 518 precomputed features for each song.
        It'll load in as a pandas DataFrame with 8,000 rows and 518 columns.
        Probably *not* the features you want to pass to the densenet. 

    genre_csv: contains genre data. I didn't use this in my experiments.

    distance_matrix: This is an 8,000 x 8,000 distance matrix between
        all the songs, used for score calculation. It's saved with nans
        along the main diagonal; I've been using np.fill_diagonal(D,0) to 
        replace them with 0s.

    meta_filename = this has the metadata for all of the songs.  This includes
        the target vector and train / valid / test assignments.
    """
    # Load in distance matrix

    if data.lower() == "std":
        tensor_name="all_data_tensor_power_db.npy"
    elif data.lower() == "big":
        tensor_name="all_data_tensor_power_db_0-5_stride.npy"
    elif data.lower() == "lsr":
        tensor_name="all_data_tensor_power_db_SR22K.npy"        

    meta = pd.read_csv(_meta_filename, index_col="track_id")
    
    # Create integer label vector

    y_labels, _ = pd.factorize(meta.track_genre_top.values)

    print(f"Loading data from tensor stored at: {_root_folder}tensors/{tensor_name}")
     
    input_tensor = np.load(f"{_root_folder}tensors/{tensor_name}")
    
    examples_per_song = int(input_tensor.shape[0] / len(y_labels))

    y_repeat = np.repeat(y_labels, examples_per_song)
    y_enc = OneHotEncoder(handle_unknown='ignore')

    y = y_enc.fit_transform(y_repeat.reshape(-1, 1))

    training_set, validation_set, test_set = get_boolean_selection_vectors(examples_per_song)

    # Use both validation and train sets for model training
    if mode.lower()=="test":
        train_and_valid_set = np.logical_or(training_set, validation_set)

        X_train_raw, y_train = input_tensor[train_and_valid_set], y[train_and_valid_set]
        X_eval_raw, y_eval = input_tensor[test_set], y[test_set]

        # use train and valid examples for scoring
        samples_for_scoring = train_and_valid_set

    # Train on training set, test on validation set
    else:
        X_train_raw, y_train = input_tensor[training_set], y[training_set]
        X_eval_raw, y_eval = input_tensor[validation_set], y[validation_set]

        # use only training examples for scoring
        samples_for_scoring = training_set

    if calc_scores:
        _distance_matrix_filename=Path(f"{_root_folder}distances/distance_matrix.npy")
            
        D = np.load(_distance_matrix_filename)
        np.fill_diagonal(D,0)
        
        train_song = samples_for_scoring[::examples_per_song]
        y_song = np.argmax(y_train.A, axis=1)[::examples_per_song]

        sample_weights = get_sample_weights(
            train_song, y_song, D
        )

        sample_weights *= (sample_weights.shape[0] / np.sum(sample_weights))

        sample_weights = np.repeat(sample_weights, examples_per_song)

    else: 
        sample_weights = np.ones(X_train_raw.shape[0])

    if remove_nan:
        # Check each data tensor for nan. If 
        # nans are found, remove examples with 
        # nan entries.

        # X_train:
        if np.any(np.isnan(X_train_raw)):
            X_train_raw_nan_examples = np.isnan(X_train_raw.reshape(
                (X_train_raw.shape[0], np.prod(X_train_raw.shape[1:]))
            )).any(axis=1)

            X_train_raw = X_train_raw[np.invert(X_train_raw_nan_examples)]
            y_train = y_train[np.invert(X_train_raw_nan_examples)]

            sample_weights = sample_weights[np.invert(X_train_raw_nan_examples)]

        # X_eval:
        if np.any(np.isnan(X_eval_raw)):
            X_eval_raw_nan_examples = np.isnan(X_eval_raw.reshape(
                (X_eval_raw.shape[0], np.prod(X_eval_raw.shape[1:]))
            )).any(axis=1)

            X_eval_raw = X_eval_raw[np.invert(X_eval_raw_nan_examples)]
            y_eval = y_eval[np.invert(X_eval_raw_nan_examples)]

        
    if scale.lower() == "standard":
        scaler = StandardScaler()

        X_train = scaler.fit_transform(
            X_train_raw.reshape((-1, 
                X_train_raw.shape[2])
            )
        )
        X_train = X_train.reshape(X_train_raw.shape)

        X_eval = scaler.transform(
            X_eval_raw.reshape((-1,
                X_eval_raw.shape[2])
            )   
        )
      
        X_eval = X_eval.reshape(X_eval_raw.shape)

    elif scale.lower() == "minmax":

        # Some entries are a tiny bit bigger than zero
        X_train_raw[X_train_raw > 0] = 0.0

        scaler = MinMaxScaler(feature_range=(0,1))

        X_train_raw_shape = X_train_raw.shape
        X_train = scaler.fit_transform(X_train_raw.reshape(-1,1))
        X_train = np.reshape(X_train, X_train_raw_shape)

        X_eval_raw_shape = X_eval_raw.shape
        X_eval = scaler.transform(X_eval_raw.reshape(-1,1))
        X_eval = np.reshape(X_eval, X_eval_raw_shape)

    else:
        X_train, X_eval = X_train_raw, X_eval_raw

    return X_train, X_eval, y_train.A, y_eval.A, sample_weights
