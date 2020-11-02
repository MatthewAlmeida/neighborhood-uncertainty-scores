import numpy as np 
from scipy.stats import mode


def get_voting_predictions(
    model, X, y, 
    samples_per_song
):
    """
    We use this function to calculate the 'voting' predictions:
    when each time window's individual prediction is used to cast a 
    'vote' as to which class the entire song should be predicted as.
    The class with the maximum number of votes is considered the 
    prediction and performance is measured accordingly.
    """
    probabilites = model.predict(X, verbose=0)

    y_true = np.argmax(y, axis=1)
    y_pred = probabilites.argmax(axis=1)

    predictions_per_song = y_pred.reshape((-1, samples_per_song))

    y_true_song = y_true[::samples_per_song]
    y_pred_song = np.zeros_like(y_true_song, dtype=int)

    for i, _ in enumerate(y_pred_song):
        y_pred_song[i] = mode(predictions_per_song[i])[0][0]

    return y_true_song, y_pred_song