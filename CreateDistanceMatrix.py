# pylint: disable=E0401

# loading topic-files
import json

# import numpy
import numpy as np
# Word2Vec
from gensim.models import KeyedVectors
# Earth Movers Distance
from pyemd import emd
# cost Function
from scipy.spatial.distance import pdist, squareform

# load word2vec emeddings: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
wvmodel = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin.gz', binary=True, limit=10000)


def TopicDistance(topicA, topicB):
    """
    calculate the Word Mover's Distance between two topics considering each words frequency
    """

    # extract vocab
    vocab = set([w for w in topicA if w in wvmodel] + [w for w in topicB if w in wvmodel])

    # create nBOW for each topic
    nBOW_A = np.array([0 if w not in topicA else topicA[w] for w in vocab], dtype=np.float64)
    nBOW_B = np.array([0 if w not in topicB else topicB[w] for w in vocab], dtype=np.float64)
    # build embedding distance Matrix
    embeddings = [wvmodel[w] for w in vocab]
    D_Mat = squareform(pdist(embeddings, 'euclidean')).astype(np.float64)
    # solve earth movers distance
    wmd = emd(nBOW_A, nBOW_B, D_Mat)

    # return 
    return wmd


def calculateDistanceMatrix(topics):
    """
    calculates pairwise distances of topics
    resulting Matrix not in squareform! (use scipy.spatial.distance.sqaureform to achive squareMatrix)
    """

    # make topics-list 2-dimensional
    topics = np.array(topics).reshape(-1, 1)
    # calcualte distance between two topics
    wmdist = lambda A,B: TopicDistance(A[0], B[0])
    # return matrix containing pairwise distances
    return pdist(topics, wmdist)


if __name__ == '__main__':

    source = "data/MyTopics/result_topics_1.txt"
    target = "data/topicDistanceMatrix"

    # read topics
    with open(source, 'r') as f:
        topics = json.load(f)

    # build topic distance Matrix
    distances = calculateDistanceMatrix(topics)
    # save topic Distance Matrix
    np.save(target, distances)