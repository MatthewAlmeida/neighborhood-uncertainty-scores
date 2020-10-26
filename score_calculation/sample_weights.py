"""
sample_weights.py

Get sample weights from a vector of border scores (same as neighborhood
scores, we changed the names a few times).
"""

import numpy as np

def get_weight_vector_from_scores(scores, alpha, beta, gamma, eta):
    """
    Convert neighborhood scores to sample weights. Parameters are 
    named for the variables in the paper.

    scores: 1D np.array containing the scores for each point.
        (for example, shape should be (6400,) for the training set)

    alpha: determines the slope of the logistic function, basically
        how quickly we change from upweighting to downweighting points.
        I almost always leave this at 1.

    beta: determines where the center of the logistic curve is; this 
        is kind of the "pivot", where points with lower scores are 
        upweighted and higher scores downweighted. Usually set this 
        to be the median of the nonzero scores (I believe FMA actually
        has no points with score 0).

    gamma: helps determine the maximum sample weight. The maximum weight
        will be set to (gamma + eta).

    eta: the entire function is increased by this amount. This determines 
        the minimum sample weight and, when summed with gamma, the maximum.

    A typical range for the sample weights might be [0.25, 2.0], corresponding
    to eta = 0.25 and gamma = 1.75. We find that very low eta (0, for example) 
    effectively shrinks the dataset, leading to lower performance, and very 
    high gamma (4 to 5 or more) puts too much emphasis on the points that
    are not noisy.
    """ 

    g = np.copy(scores)
    
    # There should be no negatives, but catch 
    # them here in case.
    g[g<0] = 0
    
    # Same here - the score function protects against
    # possible nan values, but we catch them here in 
    # case
    g[np.isnan(g)] = 0
    
    return (gamma / (1 + np.exp(-1*alpha*(-1*g + beta)))) + eta