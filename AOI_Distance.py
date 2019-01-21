# pylint: disable=E0401

# import numpy
import numpy as np
# EarthMoversDistance
from pyemd import emd
# make distanceMatrix square
from scipy.spatial.distance import squareform

# load topicDistances
topicDistMatrix = squareform(np.load("data/topicDistanceMatrix.npy"))


def AOI_Distance(AOI_A, AOI_B):
    """
    calculate distance between two Areas of Interest building on distances between topics
    """

    # total number of topics
    n = topicDistMatrix.shape[0]

    # extract distributions
    dist_A = np.zeros(n, dtype=np.float64)
    dist_A[list(AOI_A)] = list(AOI_A.values())

    dist_B = np.zeros(n, dtype=np.float64)
    dist_B[list(AOI_B)] = list(AOI_B.values())

    # solving earth movers distance
    distance = emd(dist_A, dist_B, topicDistMatrix)

    # return 
    return distance

if __name__ == '__main__':

    # sample areas of interest where key: topic-Index and value: distribution
    AOI_A = {0: 0.2, 3: 0.5, 4: 0.1, 7: 0.2}
    AOI_B = {1: 0.1, 3: 0.7, 6: 0.2}

    distance = AOI_Distance(AOI_A, AOI_B)
    print(distance)

