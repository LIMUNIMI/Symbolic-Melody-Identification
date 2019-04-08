# -*- coding: utf-8 -*
import glob
import numpy as np
import os

import extra.data_handling.midi as midi
from extra.utils.os_utils import save_pyc_bz


def get_score_from_midi(fn, flatten=True):
    """Read a MIDI file and return pitch, onset, duration, velocity, MIDI
    channel, and MIDI track number. Onset and duration are specified
    in quarter beats. The MIDI notes in each track of the MIDI file
    are concatenated. If `flatten' is True, the notes are ordered by
    onset, rather than by MIDI track number.

    Parameters
    ----------
    fn : str
      MIDI filename

    flatten : bool (default: True)

      if True, order
    Returns
    -------
    note_info : structured array
        Array containing the score information. Each row corresponds to a note.
        The fields are (`pitch`, `onset`, `duration`, `soprano`).
        `pitch` is the MIDI pitch of the note.
        `onset` is the score position (in beats) of the note.
        `duration` is the duration of a note (in beats).
        `channel` indicates the channel of the MIDI track
        `track` indicates the track number

    """

    # create a MidiFile object from an existing midi file
    m = midi.MidiFile(fn)

    # convert the object to type 0 (by merging all tracks into a single track)
    # if flatten:
    #     m = midi.convert_midi_to_type_0(m)

    div = float(m.header.time_division)

    note_information = []

    for track_nr in range(len(m.tracks)):
        note_inf_track = np.array([(n.note, n.onset / div,
                                    n.duration / div, n.velocity,
                                    n.channel,
                                    track_nr)
                                   for n in m.tracks[track_nr].notes],
                                  dtype=[('pitch', np.int),
                                         ('onset', np.float),
                                         ('duration', np.float),
                                         ('velocity', np.int),
                                         ('channel', np.int),
                                         ('track', np.int)])
        note_information.append(note_inf_track)

    note_information = np.hstack(note_information)

    if flatten:
        note_information = note_information[
            np.argsort(note_information['onset'])]

    return note_information


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser('Generate numpy dataset')

    parser.add_argument('match_dir',
                        help='Directory containing the midi files')

    parser.add_argument('--outdir',
                        help='Directory for storing the generated data',
                        default=None)

    args = parser.parse_args()

    # Default output directory
    if args.outdir is None:
        args.outdir = os.path.join(os.path.dirname(args.match_dir), 'midi')

    # Generate if it doesn't exist
    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    # Get all match files
    all_fns = glob.glob(os.path.join(args.mid_dir, '*.mid'))

    for fn in all_fns:
        # Name of the piece
        name = os.path.basename(fn).replace('.mid', '')
        # Generate array
        note_info = get_score_from_midi(fn)

        # Out path
        out_fn = os.path.join(args.outdir, name + '.pyc.bz')
        print 'Exporting piece to {0}'.format(out_fn)
        # Export file
        save_pyc_bz(note_info, out_fn)
