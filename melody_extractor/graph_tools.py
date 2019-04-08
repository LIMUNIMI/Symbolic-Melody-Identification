""" this uses cnn_trainer.py and Dijkstra algorithm to identify a melody in
scores in """

import time
import math

import numpy as np
import sklearn.metrics
from scipy.sparse import csgraph
from sklearn.model_selection import train_test_split

import misc_tools
import settings

try:
    import cPickle as pickle
except Exception:
    import pickle

FINAL_VALUE = -0.5


def compute_prob(note, prob_map, THRESHOLD):
    """
    compute the probability of *note* referred to *prob_map*.
    If the probability is higher than *THRESHOLD*, than the cost will
    be > 0, otherwise it will be 0.
    """
    pitch, onset, offset, ismelody = note
    m = prob_map[pitch, onset: offset]

    if m.shape[0] > 2 and settings.OUTLIERS_ON_PROB:
        m = m[modified_z_score(m)]

    if settings.AVERAGE:
        p = m.mean()
    else:
        p = np.median(m)
    if p < THRESHOLD:
        return -0.5
    else:
        return p


def build_graph_matrix(notelist, prob_map, THRESHOLD):
    """
    Returns a 2D array containing the matrix relative to the branch costs
    in the graph. For each note A in *notelist*, it creates branches to the
    notes N_i so that:
        1) onset(N_k) == onset(N_l) for each k, l
        2) onset(N_i) == min{onset(Y)} where Y: onset(Y) > offset(A), cost(Y) =
        1 - probability(Y) <= *THRESHOLD*} for each i

    This also adds two new virtual notes representing the first and the last
    note.
    """

    out = np.full((len(notelist) + 2, len(notelist) + 2),
                  np.inf,
                  dtype=settings.floatX)

    last_onset = notelist[0][0]

    # initialize the starting virtual note
    FOUND_NEXT_NOTE = False
    for i, note_i in enumerate(notelist, start=1):

        pitch_i, onset_i, offset_i, melody_i = note_i

        if onset_i > last_onset and FOUND_NEXT_NOTE:
            # we have found a note in the previous onset
            break

        cost_i = -compute_prob(note_i, prob_map, THRESHOLD)
        if cost_i > 0:
            continue
        else:
            FOUND_NEXT_NOTE = True
            out[0, i] = cost_i
            last_onset = onset_i

    for i, note_i in enumerate(notelist):

        pitch_i, onset_i, offset_i, melody_i = note_i
        FOUND_NEXT_NOTE = False

        for j, note_j in enumerate(notelist[i + 1:], start=1):

            pitch_j, onset_j, offset_j, melody_j = note_j

            if onset_j < offset_i:
                continue
            elif FOUND_NEXT_NOTE and notelist[i + j - 1][0] < onset_j:
                break

            cost_j = -compute_prob(note_j, prob_map, THRESHOLD)
            if cost_j > 0:
                continue
            else:
                FOUND_NEXT_NOTE = True
                # i + 1 because we have added a virtual note
                out[(i + 1), (i + 1) + j] = cost_j
                last_onset = onset_j  # this is the last note reachable

        if not FOUND_NEXT_NOTE:
            # let's jump to the last virtual state
            out[(i + 1), -1] = FINAL_VALUE

    # making the last notes pointing to the ending virtual note
    # is this required??
    for i, note_i in enumerate(reversed(notelist), start=2):
        if note_i[1] < last_onset:
            break
        elif note_i[1] > last_onset:
            continue
        else:
            out[-i, -1] = FINAL_VALUE

    return out


def _check(notelist, pianoroll_prob):
    """
    Just for debugging
    """
    WIN_WIDTH = int(settings.ARG_DEFAULT['win_w'])
    EPS = misc_tools.EPS(0)
    for j, (onset, offset, pitch) in enumerate(notelist):
        flag = False
        if pianoroll_prob[pitch, onset] < EPS:
            for i in range(WIN_WIDTH):
                if pianoroll_prob[pitch, onset - i] >= EPS:
                    print("you're wrong of -" + str(i) +
                          " for onset of note " + str(j))
                    flag = True
                    break
                if pianoroll_prob[pitch, onset + i] >= EPS:
                    print("you're wrong of +" + str(i) +
                          " for onset of note " + str(j))
                    flag = True
                    break
        elif pianoroll_prob[pitch, offset - 1] < EPS:
            for i in range(WIN_WIDTH):
                if pianoroll_prob[pitch, offset - 1 - i] >= EPS:
                    print("you're wrong of -" + str(i) +
                          " for offset of note " + str(j))
                    flag = True
                    break
                if pianoroll_prob[pitch, offset - 1 + i] >= EPS:
                    print("you're wrong of +" + str(i) +
                          " for offset of note " + str(j))
                    flag = True
                    break
        else:
            for i in range(onset, offset):
                if pianoroll_prob[pitch, i] < EPS:
                    print("note " + str(j) + " has some internal values set to 0")
                    flag = True
                    break
        if flag:
            return 1
        # if not flag:
            # print("note " + str(j) + " is correct")
    return 0


def modified_z_score(ys):
    """
    PARAMETERS :
    ------------
        list-like object, usually 1D np.array

    RETURN :
    --------
        a new 1D np.array containing the indices of elements not ouliers


    stolen from http://colingorrie.github.io/outlier-detection.html
    """
    threshold = 3.5

    median_y = np.median(ys)
    median_absolute_deviation_y = np.median([np.abs(y - median_y) for y in ys])
    modified_z_scores = [0.6745 * (y - median_y) / median_absolute_deviation_y
                         for y in ys]
    return np.where(np.abs(modified_z_scores) < threshold)[0]


def iqr(ys):
    """
    PARAMETERS :
    ------------
        list-like object, usually 1D np.array

    RETURN :
    --------
        a new 1D np.array containing the indices of elements not ouliers

    stolen from http://colingorrie.github.io/outlier-detection.html
    """
    quartile_1, quartile_3 = np.percentile(ys, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((ys < upper_bound) | (ys > lower_bound))[0]


def set_threshold(arr, CLUSTERING='single'):
    print("starting clustering")
    arr = arr.reshape(-1)
    arr = arr[arr > settings.MIN_TH]
    N_CLUSTER = 2
    target_cluster = 1
    print("max, min: ", arr.max(), arr.min())

    arr = arr[iqr(arr)]

    if CLUSTERING == 'kmeans':
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=N_CLUSTER,
                        init=np.array([settings.MIN_TH, arr.max()]).reshape(-1, 1))

        labels = kmeans.fit_predict(arr.reshape(-1, 1))
    else:
        import fastcluster
        from scipy.cluster.hierarchy import fcluster
        from scipy.spatial.distance import pdist

        Z = pdist(arr.reshape(-1, 1))
        if CLUSTERING == 'single':
            X = fastcluster.single(Z)
        elif CLUSTERING == 'average':
            X = fastcluster.average(Z)
        elif CLUSTERING == 'centroid':
            X = fastcluster.centroid(Z)
        else:
            return settings.THRESHOLD

        labels = N_CLUSTER - fcluster(X, N_CLUSTER, 'maxclust')

    # setting 0 for the minimum cluster
    # np.ma.masked_array returns only values where the mask is 0
    index = {}
    for i, l in enumerate(labels):
        index[l] = arr[i]
        if len(index.keys()) == N_CLUSTER:
            break

    index = sorted(index.items(), key=lambda kv: kv[1]) # list of tuples sorted by values
    target_label = index[target_cluster - 1][0] # the label of the desired cluster
    th = np.max(arr[np.flatnonzero(labels == target_label)]) # max of the down cluster
    print("found threshold: " + str(th))
    # print(str(np.ma.masked_array(arr, 1 - labels).min()))

    return th


def polyphonic_part(notelist, pianoroll_prob, THRESHOLD):
    """
    Returns a list of int: 1 if the note at that index in *notelist* has a
    probability > *THRESHOLD*, 0 otherwise.
    """

    predicted_labels = []

    for note in notelist:
        c = compute_prob(note, pianoroll_prob, THRESHOLD)
        if np.isnan(c):
            # don't know why this happens, we had already discarded nans...
            predicted_labels.append(0)
        else:
            predicted_labels.append(int(math.ceil(c)))

    return predicted_labels


def monophonic_part(notelist, pianoroll_prob, THRESHOLD):
    """
    Compute a strictly monophonic part by using the shortest path algorithm
    specified in `settings`

    RETURNS :
        a tuple containing :
            list(int) : the predicted labels
            list(int) : the melody indices
    """
    # compute the graph matrix
    graph = build_graph_matrix(notelist, pianoroll_prob, THRESHOLD)

    # compute the minimum paths
    dist_matrix, predecessors = csgraph.shortest_path(graph, method=settings.PATH_METHOD,
                                                      directed=True,
                                                      indices=[0],
                                                      return_predecessors=True)
    # building the predicted array label
    last = predecessors[0, -1]
    predicted_labels = [0 for j in range(len(notelist) + 2)]
    melody_indices = []
    while last != -9999:
        predicted_labels[last] = 1
        melody_indices.append(last)
        last = predecessors[0, last]

    predicted_labels = predicted_labels[1:-1]
    return predicted_labels, melody_indices


def predict_labels(pianoroll_prob, in_notelist):
    """
        Compute notes in the solo part according to the input notelist
        and a pianoroll probability distribution.

        PARAMETERS :
        pianoroll_prob : 2d np.array
            the pianoroll distribution
        in_notelist : 2d np.array
            the input list of notes as returned by
            `utils.pianoroll_utils.make_pianorolls`

        RETURNS :
        a tuple of 1D arrays :
            true labels according to `in_notelist` (1 where there is a 'solo
                part', 0 where there isn't)
            predicted labels according to `in_notelist`
    """
    # ordering notelist by onset:
    notelist = in_notelist[in_notelist[:, 1].argsort()]

    # changing all nan to 2 * EPS(0)
    for i in np.nditer(pianoroll_prob, op_flags=['readwrite']):
        if np.isnan(i):
            i[...] = 2 * misc_tools.EPS(0)
    # np.nan_to_num(pianoroll_prob, copy=False)

    # looking for the first non empty column
    s = pianoroll_prob.sum(axis=0).nonzero()[0]
    # first column with non zero values minus first onset
    pad_length = s[0] - in_notelist[0][1]
    notelist = [(pitch, onset + pad_length, offset + pad_length, ismelody)
                for pitch, onset, offset, ismelody in in_notelist]

    # notelist has no more the ground-truth, so we are using in_notelist
    true_labels = zip(*in_notelist)[-1]

    THRESHOLD = settings.THRESHOLD
    if settings.CLUSTERING != 'None':
        THRESHOLD = set_threshold(
            pianoroll_prob, CLUSTERING=settings.CLUSTERING)

    if settings.MONOPHONIC:
        # compute the graph matrix
        predicted_labels = monophonic_part(
            notelist, pianoroll_prob, THRESHOLD)[0]

    else:
        predicted_labels = polyphonic_part(
            notelist, pianoroll_prob, THRESHOLD)
    return np.array(true_labels), np.array(predicted_labels)


def test_shortest_path(test_notelists, predictions, pieces_indices=None, OUT_FILE=None):
    """ This build a graph starting from *test_notelists* and *predictions* and
    computes the minimum cost path through Dijkstra algortihm for DAG non-negative
    weighted graphs.

    It also computes Precision, Recall and F-measure for each piece in
    *test_notelists * and the avarage F_measure

    *test_notelists * must be an array-like of notelists as created by
    *misc_tools.load_files*

    RETURNS:
        a tuple of three lists containing:
            * fmeasures
            * precisions
            * recalls
        computed on pieces in the notelists in input

    """
    fmeasure_list = []
    precision_list = []
    recall_list = []
    for i in range(len(test_notelists)):
        pianoroll_prob = predictions[i]
        in_notelist = test_notelists[i]
        true_labels, predicted_labels = predict_labels(
            pianoroll_prob, in_notelist)

        # compute fmeasure, precision and recall:
        precision = sklearn.metrics.precision_score(
            true_labels, predicted_labels)
        recall = sklearn.metrics.recall_score(true_labels, predicted_labels)
        fmeasure = 2 * precision * recall / (precision + recall)
        if np.isnan(fmeasure):
            fmeasure = 0.0

        if (OUT_FILE is not None) and (pieces_indices is not None):
            OUT_FILE.write("\nPiece number: " + str(pieces_indices[i]))
            OUT_FILE.write("\nPrecision: " + str(precision))
            OUT_FILE.write("\nRecall: " + str(recall))
            OUT_FILE.write("\nF1-measure: " + str(fmeasure) + "\n")

        print("Piece number: " + str(pieces_indices[i]))
        print("Precision: " + str(precision))
        print("Recall: " + str(recall))
        print("F1-measure: " + str(fmeasure))
        print("")
        fmeasure_list.append(fmeasure)
        precision_list.append(precision)
        recall_list.append(recall)

    if (OUT_FILE is not None) and (pieces_indices is not None):
        OUT_FILE.flush()
    # print("Avarage fmeasure scores: " + str(np.mean(fmeasure_list)))
    return fmeasure_list, precision_list, recall_list
