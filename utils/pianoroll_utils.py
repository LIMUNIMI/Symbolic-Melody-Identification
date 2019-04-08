# -*- coding: utf-8 -*
"""
Functions for handling piano rolls
"""
import numpy as np
import matplotlib.pyplot as plt

import melody_extractor.settings as settings
from data_handling.parse_data import load_piece

floatX = settings.floatX


def make_pianorolls(piece, beat_div=8,
                    output_idxs=False):
    """
    Generate score and melody piano rolls.

    Parameters
    ----------
    piece : dict
        Structured array as returned by data_handling.parse_data.load_piece(...)
    beat_div : int
        Resolution for the beat (number of pixels for a beat).
        Default is 8.
    output_idxs: bool
        If `True`, returns indices of the notes in the piano roll.

    Returns
    -------
    pr_score : ndarray
        A 2D array with the piano roll of the score.
        The shape is (128, M) where M is the length of the piano roll.
    pr_melody : ndarray
        A 2D array containing the piano roll of the melody.
    idx_score : ndarray
        Indices of the notes of score in the piano roll (returned only
        if `output_idxs` is `True`)
        A list of lists is returned. Ech inner list is a note, with:
            - midi pitch value
            - starting time index
            - ending time index
            - is soprano or not (0 or 1)
    idx_score : ndarray
        Indices of the notes of the melody in the piano roll (returned only
        if `output_idxs` is `True`)
        A list of lists is returned. Ech inner list is a note, with:
            - midi pitch value
            - starting time index
            - ending time index
            - is soprano or not (0 or 1)
    """
    # Creating indices of notes not graces
    not_grace_idx = np.argwhere(piece['duration']).reshape(-1)

    # Get info from the score
    pitch = piece['pitch'][not_grace_idx]
    onset = piece['onset'][not_grace_idx]
    offset = onset + piece['duration'][not_grace_idx]
    soprano = piece['soprano'][not_grace_idx]

    # indices of the melody
    melody_idx = np.where(soprano == 1)[0]

    # note_to_pianoroll requires onset, offset and pitch
    score_info = np.column_stack([onset, offset, pitch])
    melody_info = np.column_stack([onset[melody_idx],
                                   offset[melody_idx],
                                   pitch[melody_idx]])
    # Compute piano rolls
    pr_score, idx_score = _notes_to_pianoroll(
        notes=score_info, onset_only=False,
        neighbour_pitches=-1, neighbour_beats=0,
        beat_div=beat_div,
        soprano=soprano)

    pr_melody, idx_melody = _notes_to_pianoroll(
        notes=melody_info, onset_only=False,
        min_time=np.min(onset),
        max_time=np.max(offset),
        neighbour_pitches=-1, neighbour_beats=0,
        beat_div=beat_div)

    assert pr_melody.shape == pr_score.shape

    if output_idxs:
        return pr_score, pr_melody, idx_score, idx_melody
    else:
        return pr_score, pr_melody


def get_onsetwise_pitch(pitch, onset, offset=None, melody=None):
    unique_onsets = np.unique(onset)

    unique_onset_idx = [np.where(onset == u)[0] for u in unique_onsets]

    unique_pitch_array = []

    pitch_sidx = []
    for u in unique_onset_idx:
        u_p = pitch[u]
        u_p_i = u_p.argsort()
        u_p = u_p[u_p_i]
        pitch_sidx.append(u_p_i)

        unique_pitch_array.append(u_p)

    unique_pitch_array = np.array(unique_pitch_array)

    if offset is None and melody is None:
        return unique_pitch_array, unique_onsets
    else:
        unique_offset_array = np.array(
            [offset[u[p]] for u, p in zip(unique_onset_idx, pitch_sidx)])

        return unique_pitch_array, unique_onsets, unique_offset_array


def get_pianoroll_indices(pitch, onset, offset, soprano, beat_div=8):
    """
    Get indices of piano roll corresponding to the notes in a piece

    Parameters
    ----------
    pitch : array
        Array with the MIDI pitch of each note
    onset : array
        Array with the onset time in beats for each note
    offset : array
        Array with the offset time in beats for each note
    soprano : array
        Array indicating if a note is part of the melody.

    Returns
    -------
    idx_score : array
    Array containing the indices of each note in the piano roll :
        (pitch, start_time, end_time, is_soprano)
    Where *start_time* is the index of the first column in which the note is
    playing and *end_time* is the index of the first column in which the note
    is no more playing.
    """
    idx_score = np.zeros((len(pitch), 4)).astype(int)
    idx_score[:, 0] = pitch
    idx_score[:, 1] = np.round(onset * beat_div)
    idx_score[:, 2] = np.round(offset * beat_div)
    idx_score[:, 2] = np.maximum(idx_score[:, 1] + 1,
                                 idx_score[:, 2] - 1)

    if soprano is not None:
        idx_score[:, 3] = soprano

    return idx_score


def _notes_to_pianoroll(notes, onset_only,
                        neighbour_pitches,
                        neighbour_beats,
                        beat_div,
                        min_time=None,
                        max_time=None,
                        soprano=None):

    # columns:
    ONSET = 0
    OFFSET = 1
    PITCH = 2

    if neighbour_pitches > -1:
        highest_pitch = np.max(notes[:, PITCH])
        lowest_pitch = np.min(notes[:, PITCH])
    else:
        lowest_pitch = 0
        highest_pitch = 127

    pitch_span = highest_pitch - lowest_pitch + 1

    # Get minimum and maximum time
    if min_time is None:
        min_time = np.min(notes[:, ONSET])
    if max_time is None:
        max_time = np.max(notes[:, OFFSET])

    # shift times to start at 0
    notes[:, ONSET] -= min_time - neighbour_beats
    notes[:, OFFSET] -= min_time - neighbour_beats
    if neighbour_pitches > -1:
        notes[:, PITCH] -= lowest_pitch
        notes[:, PITCH] += neighbour_pitches
    # size of the feature matrix
    if neighbour_pitches > -1:
        M = pitch_span + 2 * neighbour_pitches
    else:
        M = pitch_span
    N = int(np.ceil(beat_div * (2 * neighbour_beats + max_time - min_time)))

    pr_idxs = get_pianoroll_indices(notes[:, PITCH],
                                    notes[:, ONSET],
                                    notes[:, OFFSET],
                                    soprano,
                                    beat_div)
    pianoroll = np.zeros((M, N), dtype=floatX)

    for n in pr_idxs:
        pianoroll[n[0], n[1]:n[2]] = 1

    return pianoroll, pr_idxs


def plot_pianorolls(pr_score, pr_melody, out_fn='/tmp/pianorolls.pdf'):
    """
    Plot piano rolls

    Parameters
    ----------
    pr_score : ndarray
        2D array containing the piano roll of the score.
    pr_melody : ndarray
        2D array containing the piano roll of the melody.
    out_fn : str
        Path to save the figure. Default is `/tmp/pianorolls.pdf`.
    """
    # Initialize axes
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    pr_score = pr_score[:, ~np.all(pr_score == 0, axis=0)]
    # pr_melody = pr_melody[:, ~np.all(pr_melody == 0, axis=0)]

    # Plot piano rolls
    ax1.matshow(pr_score, aspect='auto',
                origin='lower',
                cmap='gray_r')
    ax1.set_ylabel('Pitch')
    ax2.matshow(pr_melody, aspect='auto',
                origin='lower',
                cmap='gray_r')
    ax2.set_ylabel('Pitch')
    ax2.set_xlabel('Time')

    xticks = np.arange(0, pr_score.shape[1], 128)
    yticks = np.arange(0, 128, 12)

    ax1.set_xticks(xticks)
    ax2.set_xticks(xticks)
    ax1.set_yticks(yticks)
    ax2.set_yticks(yticks)

    ax2.grid(linestyle='dashed', linewidth=0.1)
    ax1.grid(linestyle='dashed', linewidth=0.1)

    # Save figures
    plt.savefig(out_fn)
    plt.clf()
    plt.close()


def test__notes_to_pianoroll():
    import glob
    import os
    beat_div = 8

    all_fns = glob.glob(os.path.join(
        'data/solo-accompaniment-dataset/cultivated/', '*.pyc.bz'))

    for fn in all_fns:
        print fn
        # Load piece
        piece = load_piece(fn)

        # Get info from the score
        pitch = piece['pitch']
        onset = piece['onset']
        offset = piece['onset'] + piece['duration']
        soprano = piece['soprano']
        # indices of the melody
        melody_idx = np.where(piece['soprano'] == 1)[0]

        # note_to_pianoroll requires onset, offset and pitch
        score_info = np.column_stack([onset, offset, pitch])
        melody_info = np.column_stack([onset[melody_idx],
                                       offset[melody_idx],
                                       pitch[melody_idx]])
        # Compute piano rolls
        pr_score, idx_score = _notes_to_pianoroll(
            notes=score_info, onset_only=False,
            neighbour_pitches=-1, neighbour_beats=0,
            beat_div=beat_div, soprano=soprano)

        # pr_score = pr_score.todense().astype(floatX)
        pr_melody, idx_melody = _notes_to_pianoroll(
            notes=melody_info, onset_only=False,
            min_time=np.min(onset),
            max_time=np.max(offset),
            neighbour_pitches=-1, neighbour_beats=0,
            beat_div=beat_div)

        assert pr_melody.shape == pr_score.shape

        active_notes = np.hstack(
            [pr_score[n[0], n[1]:n[2]].flatten() for n in idx_score])

        print np.mean(active_notes)

        print np.all(idx_score[idx_score[:, 3] == 1, :3] == idx_melody[:, :3])


def test_make_pianorolls():
    import glob
    import os
    beat_div = 8

    all_fns = glob.glob(os.path.join(
        'data/hyper-opt/', '*.pyc.bz'))

    for fn in all_fns:
        print fn
        # Load piece
        piece = load_piece(fn)
        pr_score, pr_melody, idx_score, idx_melody = make_pianorolls(
            piece, beat_div=beat_div, output_idxs=True)

        # Check that both the melody and the score have the same information
        print np.all(idx_score[idx_score[:, 3] == 1, :3] == idx_melody[:, :3])


if __name__ == '__main__':

    fn = '../data/mozart/kv279_1.pyc.bz'
    pr_score, pr_melody, idx_score, idx_melody = make_pianorolls(
        fn, beat_div=8, output_idxs=True)
    # Generate plot
    plot_pianorolls(pr_score, pr_melody)

    print np.all(idx_score[idx_score[:, 3] == 1, :3] == idx_melody[:, :3])
