"""
border_scores.py

New function created 12-2-19 to bring neighborhood
score calculation into python 3 with proper
documentation and packaging.
"""

import numpy as np
import os


def get_border_scores_precomputed_distance(y, D, k, 
                                           use_numerator_distance=True, 
                                           use_denominator_distance=True,
                                           normalize_distance=True):
    """
    Calculates border scores using formula in section 3.3 of
    paper. border score is a pointwise calculation:

    bs = 
    
    -C * (k_yi / k) log (k_yi / k) * (k_yi / sum(d_xi)) 
    /
    - sum_j^C (k_j/k) log (k_j/k) * kj / d_j
    
    Inputs
    -------
    y: 1D numpy array container the pointwise integer 
        class labels.  Assumes labels indexed 0 ... n_labels
        without gaps.

    D: square 2D numpy array of shape (|y|, |y|) that gives
        the pointwise distances.

    k: integer neighborhood size

    Returns
    -------
    bs: 1D numpy array of length |y| containing the 
        pointwise border scores.

    """

    n_classes = np.max(y) + 1

    # argsort along each row (point), finding the closest 
    # points. Throw out column 0, which will correspond
    # to the distance to the point itself. Keep the next k
    # columns, which correspond to the next k closest points.
    #
    # neighborhoods[i] gives a vector corresponding to the
    # indices of the k closest points to point i.
    
    neighborhoods = np.argsort(D, axis=1)[:,1:k]

    # Indexing y (the label vector) with neighborhoods yields
    # an n x k matrix with each entry as the class label of that
    # point in the neighborhood.

    nbhd_classes = y[neighborhoods].squeeze()

    # Sneaky indexing here. If we declare a verital vector with
    # ascending indices, then index with that vertical vector 
    # first and then the neighborhood matrix, we can get a 
    # matrix corresponding to the neighborhood matrix but with 
    # distances instead. Newaxis is used here to make 
    # vertical_index_vector of shape (|y|, 1).

    vertical_index_vector = np.arange(y.shape[0])[:,np.newaxis]

    # Use the vertical index vector here. Numpy is amazing. 

    distances_to_neighborhood_points = D[vertical_index_vector, neighborhoods]

    # Normalize the distance, if instructed to by the neighborhood parameter.

    if normalize_distance:
        distances_to_neighborhood_points = np.true_divide(
            distances_to_neighborhood_points, 
            np.min(np.abs(distances_to_neighborhood_points), axis=1)[:, np.newaxis]
        )

    # Initialize data structures for the numerator and denominator
    numerator = np.zeros_like(y)
    denominator = np.zeros_like(y)

    # Loop through the classes calculating the score

    for c in range(n_classes):
        # Mask identifying points of the current class
        curr_class = y == c

        # Mask identifying points in neighborhoods of current class
        neighbors_of_class_c = nbhd_classes == c

        # Identify the probability of a neighbor being of current class;
        # include all neighbors plus point itself.
        count_of_neighbors_of_class_c = np.sum(neighbors_of_class_c, axis=1) + curr_class
        prob_class_c = np.divide(count_of_neighbors_of_class_c, k)

        # Calculate the log of the probability of a neighbor being of the current class.
        # Use log 0 = 0 for these purposes.
        log_prob_class_c = np.where(prob_class_c > 0, np.log2(prob_class_c), 0)

        # Caclulate the sum of the distances to the points of the current class, 
        # then divide by the number of points to yield average distance

        avg_distance_to_class_c = np.where(
            count_of_neighbors_of_class_c != 0,
            np.divide(
                np.sum(
                    np.where(
                        neighbors_of_class_c,
                        distances_to_neighborhood_points,
                        0
                    ),
                    axis=1
                ),
                count_of_neighbors_of_class_c
            ),
            0
        )
        
        avg_inv = np.divide(np.ones_like(avg_distance_to_class_c), avg_distance_to_class_c)
        avg_inv[np.invert(np.isfinite(avg_inv))] = 0

        # If a point is of the current class, add the entropy term to the numerator.
        # If not, add it to the denominator. If we are using distance, combine that 
        # in.

        if use_numerator_distance:
            numerator = np.where(curr_class,
                np.where(avg_distance_to_class_c != 0,
                    numerator + (prob_class_c * log_prob_class_c * avg_inv),
                    numerator + (prob_class_c * log_prob_class_c) 
                ),
                numerator
            )
        else:
            numerator = np.where(curr_class, 
                numerator + (prob_class_c * log_prob_class_c),
                numerator
            )

        if use_denominator_distance:
            denominator = np.where(avg_distance_to_class_c != 0,
                    denominator + (prob_class_c * log_prob_class_c * avg_inv),
                    denominator + (prob_class_c * log_prob_class_c)
            )
        else:
            denominator = denominator + (prob_class_c * log_prob_class_c)

    # Here we figure out how many classes are in each neighborhood.
    y_t = y[:,np.newaxis]
    point_and_nbhd = np.hstack((y_t, nbhd_classes))
    counts = np.array([np.unique(x).shape[0] for x in point_and_nbhd])

    # Negate numerator and denominator to make them positive.
    # Multiply the numerator by the number of classes in the neighborhood
    # to balance it with the denominator.
    numerator *= (-1 * counts)
    denominator *= -1

    # If the denominator is 0, return 0. Else, return the 
    # numerator divided by the denominator.
    bs = np.where(denominator > 0,
        np.divide(numerator, denominator),
        0    
    )

    # For some reason we can end up with -0 in the
    # final result vector. This line simply replaces
    # -0s with 0s, because both 0 == 0  and -0 == 0 
    # return True.
    bs[bs == 0] = 0

    return bs
