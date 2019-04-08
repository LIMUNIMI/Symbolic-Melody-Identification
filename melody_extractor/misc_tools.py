import os

import numpy as np
import sklearn.preprocessing
import theano.tensor as T
import utils.pianoroll_utils
from data_handling.parse_data import load_piece

import settings


def build_groups(dataset_map):
    """
    Given a *dataset_map* returns a tuple of lists :
        * a np.ndarray containing samples
        * the corresponding np.ndarray of groups ready-to-be-used in scklearn KFold
    Groups start counting from 0.
    """
    group_set = []
    groups = []
    for p, piece in enumerate(dataset_map):
        for window in piece:
            group_set.append(window)
            groups.append(p)
    return np.array(group_set), np.array(groups)


def recreate_pianorolls(array_of_windows, overlap=True):
    """
    Recreates pianorolls from an array of windows.
    *array_of_windows* MUST be a 4D array.

    If *overlapping* is True, the windows will be thought as overlapping by 50%

    RETURNS :
        a 2D array with dimensions (128, length of pianoroll)
    """

    WIN_WIDTH = array_of_windows.shape[3]
    WIN_HEIGHT = array_of_windows.shape[2]
    NUM_WIN = array_of_windows.shape[0]
    pianoroll_width = WIN_WIDTH * NUM_WIN
    if overlap:
        pianoroll_width /= 2
        pianoroll_width += WIN_WIDTH / 2

    output = np.zeros((WIN_HEIGHT, pianoroll_width))

    for i in range(NUM_WIN):
        window = array_of_windows[i][0]
        if overlap:
            output[:, i * WIN_WIDTH /
                   2: (i + 2) * WIN_WIDTH / 2] += window / 2.0
        else:
            output[:, i * WIN_WIDTH: (i + 1) * WIN_WIDTH] += window

    return output


def split_windows(array2d, WIN_WIDTH, overlap):
    """
    Proxy function for `overlapping_split` and `no_overlap_split`
    according to *overlap*.

    If WIN_WIDTH > array2d.shape[1] (the number of columns in the pianoroll),
    then it is just padded to WIN_WIDTH
    """
    if overlap:
        if WIN_WIDTH > array2d.shape[1]:
            return _no_overlap_split(array2d, WIN_WIDTH)
        else:
            return _overlapping_split(array2d, WIN_WIDTH)
    else:
        return _no_overlap_split(array2d, WIN_WIDTH)


def _no_overlap_split(array2d, WIN_WIDTH):
    """
    utility function for *load_files*

    RETURN :
        a list of non-overlapping arrays all of width *WIN_WIDTH*
        with the last one padded by zeros; if *WIN_WIDTH* is not even,
        then *None* is returned.
    """
    import math

    if WIN_WIDTH % 2 != 0:
        return None

    NUM_WIN = math.ceil(array2d.shape[1] / float(WIN_WIDTH))
    pad_len = int(NUM_WIN * WIN_WIDTH - array2d.shape[1])
    pad = np.pad(array2d, ((0, 0), (0, pad_len)),
                 'constant', constant_values=0)

    return np.array_split(pad, NUM_WIN, axis=1)


def _overlapping_split(array2d, WIN_WIDTH):
    """
    utility function for *load_files*

    RETURN :
        a list of overlapping arrays all of width *WIN_WIDTH*
        with the first and the last one padded by
        with zeros; if *WIN_WIDTH* is not even, then *None* is returned.
    """

    if WIN_WIDTH % 2 != 0:
        return None

    length = array2d.shape[1]
    split_indices = [i for i in range(WIN_WIDTH / 2, length, WIN_WIDTH / 2)]
    split_indices.append(length)
    if split_indices[-2] != length:
        split_indices.append(length)
    if len(split_indices) % 2 != 0:
        split_indices.append(length)
    splitted = []
    start_a = 0
    start_b = 0

    for i in range(0, len(split_indices), 2):
        end_a = split_indices[i]
        end_b = split_indices[i + 1]
        splitted.append(array2d[:, start_a:end_a])
        to_be_added = array2d[:, start_b:end_b]
        if to_be_added.shape[1] != 0:
            splitted.append(to_be_added)
        start_a = end_a
        start_b = end_b

    # padding
    pad_a = WIN_WIDTH - splitted[0].shape[1]
    pad_b_1 = WIN_WIDTH - splitted[-1].shape[1]
    pad_b_2 = WIN_WIDTH - splitted[-2].shape[1]

    splitted[0] = np.pad(splitted[0], [(0, 0), (pad_a, 0)], 'constant')
    splitted[-1] = np.pad(splitted[-1], [(0, 0), (0, pad_b_1)], 'constant')
    splitted[-2] = np.pad(splitted[-2], [(0, 0), (0, pad_b_2)], 'constant')

    return splitted


def load_files(path, WIN_WIDTH=settings.WIN_WIDTH, extensions=settings.FILE_EXTENSIONS, return_notelists=False, overlap=True):
    """
    Load files from path.

    If *WIN_WIDTH* is not even, it returns NONE.

    If *overlap* is True, then windows will be returned with a 50% overlap.
    Windows will be overlapped for 50% and the first and the last one will be
    padded with 0 so that pianorollLength/WIN_WIDTH == 2. If *overlap* is None,
    windows will not be overlapped and just the last one will be padded by 0.

    *extensions* is a string or a tuple of extensions for files to be loaded

    RETURNS :
    A tuple with :
        * Two 4D arrays: scores, melodies. Dimensions are: (window_index,
        1, height_of_window, width_of_window). Both the 4D arrays are pianorolls
        representations.
        * A list containing maps:
            - [[window indices for score 0] [ window indices for score 1] [...]]
    OR
        `None` if *WIN_WIDTH* is not even

    If *return_notelists* is True, then a list of notelists will be added in
    the returned tuple, one for the score and one for the melody. Each notelist
    is a 2D array in which each row contains (pitch, onset, offset,
    melody), where:
        - *pitch* is the midi pitch AND the row index in the pianoroll
        - *onset* is the column index in which the note start in the pianoroll
        - *offset* is the column index in which the note ends in the pianoroll
        - *melody* is 1 if the note is a melody note, 0 otherwise
    See *utils.pianoroll_utils.get_pianoroll_indices* for this output.
    """

    extensions = settings.FILE_EXTENSIONS
    if WIN_WIDTH % 2 != 0:
        return None

    score_list = []
    melody_list = []
    map_score_window = []
    notelist_scores = []

    file_list = []
    # recurse all directories
    print(extensions)
    for root, subdirs, files in os.walk(path):
        # take just the files with the proper extension
        file_list += [os.path.join(root, fp)
                      for fp in files if fp.endswith(extensions)]

    if len(file_list) == 0:
        raise Exception("No files found with this extension!")

    # extract random files
    if settings.DATASET_PERC < 1:
        num_files = int(settings.DATASET_PERC * len(file_list))
        np.random.seed(1987)
        file_list = np.random.choice(
            file_list, num_files, replacement=False)

    for f in sorted(file_list):
        print("I've found a new file: " + f)

        note_array = load_piece(f)
        # load pianorolls score and melody
        pr = utils.pianoroll_utils.make_pianorolls(
            note_array, output_idxs=return_notelists)
        score = pr[0]
        melody = pr[1]
        if len(pr) == 4:
            notelist_scores.append(pr[2])

        # split in windows
        score_splitted = split_windows(score,
                                       WIN_WIDTH, overlap)
        melody_splitted = split_windows(melody,
                                        WIN_WIDTH, overlap)
        # update the map
        counter = len(score_list)
        map_score_window.append(
            [c + counter for c in range(len(score_splitted))])

        # update the output list
        score_list += score_splitted
        melody_list += melody_splitted

    # reshaping output
    score_out = np.ndarray(shape=(len(score_list), 1, settings.WIN_HEIGHT, WIN_WIDTH),
                           buffer=np.array(score_list), dtype=settings.floatX)

    melody_out = np.ndarray(shape=(len(melody_list), 1, settings.WIN_HEIGHT, WIN_WIDTH),
                            buffer=np.array(melody_list), dtype=settings.floatX)

    if return_notelists:
        return score_out, melody_out, map_score_window, np.array(notelist_scores)
    else:
        return score_out, melody_out, map_score_window


def evaluate(prediction, ground_truth):
    """ INPUT: three 2D arrays
    RETURNS: true_positives, false_positives, true_negatives and false negatives
        for this prediction """
    tp = fp = tn = fn = 0

    for (pitch, timing), predicted_value in np.ndenumerate(prediction):
        true_value = ground_truth[pitch, timing]

        if predicted_value > 0.5:
            # positive...
            if true_value > 0.5:
                # ...true
                tp += 1
            else:
                # ...false
                fp += 1
        else:
            # negative...
            if true_value <= 0.5:
                # ...positive
                tn += 1
            else:
                # ...false
                fn += 1

    return float(tp), float(fp), float(tn), float(fn)


def EPS(x):
    """
        returns 10^(-15) if x is 0, otherwise returns x
    """
    if x == 0:
        return 1.e-15
    else:
        return x


def myBinarize(arr, KMEANS=True):
    """ Autodefine a threshold by using KMeans and returns a new array with
    containing 1 where arr is > threshold, 0 where it's <= threshold
    """
    from sklearn.preprocessing import binarize

    th = 0.0
    if KMEANS:
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=2, init=np.array(
            [EPS(0), 1]).reshape(-1, 1))

        arr_reshaped = arr.flatten().reshape(-1, 1)
        labels = kmeans.fit_predict(arr_reshaped)
        min = np.ma.masked_array(arr, 1 - labels).min()
        max = np.ma.masked_array(arr, labels).max()
        th = (min + max) / 2
        # print("threshold is: " + str(th))
    return binarize(arr, threshold=th, copy=False)


def myMaskArr(arr, mask, binarize=True):
    """ Apply *mask* to *arr* by taking only the maximum value per each column.
    RETURNS:
        a new np.ndarray with shape arr.shape and with at most
        just one element != 0 per each column. If *binarize* is set to True, it
        will also set to 1 each element != 0 included in the resulting array.
    """
    # building the new array
    returned = np.zeros(arr.shape, arr.dtype)
    for y, column in enumerate(np.ma.masked_array(arr, 1 - mask).T):
        # iterating over columns masked
        x = column.argmax()
        if binarize:
            returned[x, y] = 1.0
        else:
            returned[x, y] = arr[x, y]

    return returned


def myMask(input_t, mask, binarize=False, clip=True):
    """ Same of myMaskArr but with theano thensors.
    It  performs the following :
        1) masks *input_t* with *mask*, takes only value bigger than `settings.THRESHOLD`
        # no more 2) divides each entry to the maximum value in the resulting tensor
        3) if *binarize* is True, sets 1.0  - EPS(0) in the maximum value of each column
            and EPS(0) in any other position of the column
    This uses EPS(0) to avoid nans in cross-entropy loss function.
    If *clip* is True, then the returned tensor will be clipped between EPS(0) and
    1.0 - EPS(0)

    RETURNS :
        theano tensor
    """
    assert (input_t.ndim == 4 and mask.ndim == 4 and binarize) or (not binarize),\
        "input and mask MUST be 4D tensors to the end of binarization"

    masked = mask * input_t
    # masked = masked * (masked > settings.THRESHOLD)
    # masked = abs(input_t * mask)
    # # normalized = masked
    # normalized = masked / masked.max()

    if binarize:
        binarized = T.fill(masked, EPS(0))
        max_rows = masked.argmax(axis=2)
        max_cols = T.arange(masked.shape[3])
        normalized = T.set_subtensor(
            binarized[0, 0, max_rows, max_cols], 1.0 - EPS(0))
        returned = normalized

    elif clip:
        returned = masked.clip(EPS(0), 1 - EPS(0))
    else:
        returned = masked
    return returned
