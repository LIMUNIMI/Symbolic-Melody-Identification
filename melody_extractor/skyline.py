#!/usr/bin/env pytohn2
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.metrics

from data_handling import parse_data
import misc_tools
import settings


def skyline_pianorolls(input, onset=True):
    """
    Perform the skyline algorithm on *pianoroll*. This just takes the
    highest note at each time. If *onset* is True, then the original version is
    used, in which the highest pitch referred to the most recent onset is
    considered to be melody.

    Reference paper:
    A. L. Uitdenbogerd and J. Zobel, "Melodic matching techniques for large
    music databases," in Proceedings of the 7th ACM International Conference on
    Multimedia '99, Orlando, FL, USA, October 30 - November 5, 1999, Part 1.,
    1999, pp. 57-66.

    RETURNS:
        a new array of shape *pianoroll.shape* containing the resulting pianoroll
    """

    pianoroll = np.array(input, dtype=misc_tools.floatX)
    returned = np.zeros(pianoroll.shape, pianoroll.dtype)

    for y, col in enumerate(pianoroll.T):
        # iterating over columns
        backup = False
        for x, v in enumerate(col):
            # iterating over pitches
            if v != 0:
                if onset:
                    if pianoroll[x, y - 1] == 0:
                        # new onset at highest pitch
                        returned[x, y] = v
                        backup = False
                        break
                    elif not backup:
                        # this is the highest value coming from a previous onset,
                        # store this value and add it after having parsed the whole
                        # column
                        backup = (x, y, v)
                        # N.B. now bool(backup) == True
                else:
                    returned[x, y] = v
                    break
        if backup:
            # add the highest value coming from a previous onset
            returned[backup[0], backup[1]] = backup[2]
            backup = False

    return returned


def test_skyline_pianorolls(PATH=settings.DATA_PATH):
    import os
    from data_handling import parse_data

    # recurse all directories
    dataset = []
    for root, subdirs, files in os.walk(PATH):

        # for each file with extension '.bz2'
        for f in files:

            if f[-3:] == ".bz":
                new_file = os.path.join(root, f)

                print("I've found a new file: " + new_file)
                # load pianorolls score and melody
                score, melody = parse_data.make_pianorolls(new_file)
                dataset.append((score, melody))

    train_set, test_set = train_test_split(
        dataset, test_size=0.20, random_state=42)

    overall_sk_tp = overall_sk_fp = overall_sk_tn = overall_sk_fn = 0
    overall_hp_tp = overall_hp_fp = overall_hp_tn = overall_hp_fn = 0
    avarage_pieces_sk = []
    avarage_pieces_hp = []

    for score, melody in test_set:
        sk = skyline_pianorolls(score)
        results = misc_tools.evaluate(sk, melody)
        overall_sk_tp += results[0]
        overall_sk_fp += results[1]
        overall_sk_tn += results[2]
        overall_sk_fn += results[3]
        p = results[0] / misc_tools.EPS(results[0] + results[1])
        r = results[0] / misc_tools.EPS(results[0] + results[3])
        f = 2 * r * p / misc_tools.EPS(p + r)
        avarage_pieces_sk.append((p, r, f))

        hp = skyline_pianorolls(score, onset=False)
        results = misc_tools.evaluate(hp, melody)
        overall_hp_tp += results[0]
        overall_hp_fp += results[1]
        overall_hp_tn += results[2]
        overall_hp_fn += results[3]
        p = results[0] / misc_tools.EPS(results[0] + results[1])
        r = results[0] / misc_tools.EPS(results[0] + results[3])
        f = 2 * r * p / misc_tools.EPS(p + r)
        avarage_pieces_hp.append((p, r, f))
        # parse_data.plot_pianorolls(
        #     score, sk, out_fn=f + "_skyline.pdf")
        # parse_data.plot_pianorolls(
        #     score, hp, out_fn=f + "_highestpitch.pdf")
    print("Final Results Skyline:")
    print("True positives: " + str(overall_sk_tp))
    print("False positives: " + str(overall_sk_fp))
    print("True negatives: " + str(overall_sk_tn))
    print("False negatives: " + str(overall_sk_fn))
    p = overall_sk_tp / misc_tools.EPS(overall_sk_tp + overall_sk_fp)
    r = overall_sk_tp / misc_tools.EPS(overall_sk_tp + overall_sk_fn)
    print("Precision: " + str(p))
    print("Recall: " + str(r))
    print("Fmeasures: " + str(2 * r * p / misc_tools.EPS(p + r)))
    print("Avarage piece precision: " + str(np.mean(avarage_pieces_sk[0])))
    print("Avarage piece recall: " + str(np.mean(avarage_pieces_sk[1])))
    print("Avarage piece fmeasure: " + str(np.mean(avarage_pieces_sk[2])))
    print()
    print("Final Results Highest Pitch:")
    print("True positives: " + str(overall_hp_tp))
    print("False positives: " + str(overall_hp_fp))
    print("True negatives: " + str(overall_hp_tn))
    print("False negatives: " + str(overall_hp_fn))
    p = overall_hp_tp / misc_tools.EPS(overall_hp_tp + overall_hp_fp)
    r = overall_hp_tp / misc_tools.EPS(overall_hp_tp + overall_hp_fn)
    print("Precision: " + str(p))
    print("Recall: " + str(r))
    print("Fmeasures: " + str(2 * r * p / misc_tools.EPS(p + r)))
    print("Avarage piece precision: " + str(np.mean(avarage_pieces_hp[0])))
    print("Avarage piece recall: " + str(np.mean(avarage_pieces_hp[1])))
    print("Avarage piece fmeasure: " + str(np.mean(avarage_pieces_hp[2])))


def my_skyline_notelists(notelist):
    """
    perform a variation a the skyline algorithm by taking always the highest pitch
    at each time.
    *notelist* must be in the form returned by misc_tools.load_files

    RETURNS :
        the list of predicted labels, where 1 is for melody note and 0 is for
        accompaniment
    """
    # ordering notelist by onset
    notelist = sorted(notelist, key=lambda x: x[1])

    predicted_label = [0 for n in range(len(notelist))]
    previous_onset = 99999999999  # the first time is not a new onset
    last_melody_offset = 0
    highest_pitch = 0
    melody_index = 0
    last_melody_pitch = 0
    for i, (pitch, onset, offset, ismelody) in enumerate(notelist):
        if pitch > highest_pitch:
            # look for the highest pitch among notes at this offset
            highest_pitch = pitch
            melody_index = i
        elif onset > previous_onset:
            # this is a new onset:
            # test if among notes at the previous onset there is a melody note
            if highest_pitch > last_melody_pitch or previous_onset >= last_melody_offset:
                # mark the new melody note
                predicted_label[melody_index] = 1
                last_melody_offset = notelist[melody_index][2]
                last_melody_pitch = notelist[melody_index][0]
            highest_pitch = 0
        previous_onset = onset
    return predicted_label


def skyline_notelists(notelist):
    """
    performs the skyline algorithm in its original formulation over
    the *notelist* in input.
    *notelist* is in the form returned by misc_tools.load_files

    Reference paper:
    A. L. Uitdenbogerd and J. Zobel, "Melodic matching techniques for large
    music databases," in Proceedings of the 7th ACM International Conference on
    Multimedia '99, Orlando, FL, USA, October 30 - November 5, 1999, Part 1.,
    1999, pp. 57-66.

    RETURNS :
        the list of predicted labels, where 1 is for melody note and 0 is for
        accompaniment
    """
    # ordering notelist by onset
    notelist = sorted(notelist, key=lambda x: x[1])

    predicted_label = [0 for n in range(len(notelist))]
    previous_onset = 99999999999  # the first time is not a new onset
    highest_pitch = 0
    melody_index = 0
    for i, (pitch, onset, offset, ismelody) in enumerate(notelist):
        # take all notes at this onset
        if onset > previous_onset:
            # this is a new onset
            predicted_label[melody_index] = 1
            highest_pitch = pitch
            melody_index = i
        elif pitch > highest_pitch:
            # chose the highest pitch
            highest_pitch = pitch
            melody_index = i
        previous_onset = onset
    return predicted_label


def test_skyline_notelists(PATH=settings.DATA_PATH, variation=False):
    """
    This test the skyline algorithm on the whole dataset contained in *PATH*.
    if *variation* is True, then *my_skyline_notelists* is used, otherwise
    *skyline_notelists* is used.
    """
    X, Y, map_sw, notelists = misc_tools.load_files(
        PATH, 128, return_notelists=True)
    del X, Y, map_sw

    fmeasure_list = []
    precision_list = []
    recall_list = []
    for i, notelist in enumerate(notelists):
        if variation:
            predicted_labels = my_skyline_notelists(notelist)
        else:
            predicted_labels = skyline_notelists(notelist)
        true_labels = zip(*notelist)[-1]
        precision = sklearn.metrics.precision_score(
            true_labels, predicted_labels)
        recall = sklearn.metrics.recall_score(true_labels, predicted_labels)
        fmeasure = 2 * precision * recall / (precision + recall)
        print("Piece number: " + str(i))
        print("Precision: " + str(precision))
        print("Recall: " + str(recall))
        print("F1-measure: " + str(fmeasure))
        print("")
        fmeasure_list.append(fmeasure)
        precision_list.append(precision)
        recall_list.append(recall)

    print("Average precision: " + str(np.mean(precision_list)))
    print("Average recall: " + str(np.mean(recall_list)))
    print("Average fmeasure: " + str(np.mean(fmeasure_list)))


if __name__ == '__main__':
    # print("Testing with pianorolls...")
    # print("__________________________")
    # test_skyline_pianorolls()
    DATA_PATH = settings.DATA_PATH
    import sys
    if len(sys.argv) > 1:
        DATA_PATH = sys.argv[1]
        print("Using data path: " + DATA_PATH)

    print("Testing with notelists...")
    print("__________________________")
    test_skyline_notelists(PATH=DATA_PATH, variation=False)
    # print("")
    # print("And now the variation...")
    # print("__________________________")
    # test_skyline_notelists(variation=True)
