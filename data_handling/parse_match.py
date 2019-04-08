# -*- coding: utf-8 -*
import glob
import numpy as np
import os

import extra.data_handling.matchfile as matchfile
from extra.utils.os_utils import save_pyc_bz


def get_score_from_match(fn):
    """
    Load a score from a matchfile

    Parameters
    ----------
    fn : str
        Path to the score in match format

    Returns
    -------
    note_info : structured array
        Array containing the score information. Each row corresponds to a note.
        The fields are (`pitch`, `onset`, `duration`, `soprano`).
        `pitch` is the MIDI pitch of the note.
        `onset` is the score position (in beats) of the note.
        `duration` is the duration of a note (in beats).
        `soprano` is a boolean indicating whether the current note is part
         of the main melody.
    """
    # Load matchfile
    mf = matchfile.MatchFile(fn)

    # initialize list to hold note information
    note_info = []
    for snote, note in mf.note_pairs:
        # Note name
        step = str(snote.NoteName).upper()
        # Accidental
        modifier = str(snote.Modifier)
        # Octave
        octave = snote.Octave

        # MIDI pitch
        pitch = matchfile.pitch_name_2_midi_PC(modifier, step, octave)[0]

        # Onset in beats
        onset_b = snote.OnsetInBeats
        # Offset in beats
        offset_b = snote.OffsetInBeats

        # Get melody notes
        soprano = 0
        if 's' in snote.ScoreAttributesList:
            soprano = 1

        note_info.append((pitch, onset_b, offset_b - onset_b, soprano))

    # Output as structured array
    note_info = np.array(note_info,
                         dtype=[('pitch', 'i4'),
                                ('onset', 'f4'),
                                ('duration', 'f4'),
                                ('soprano', 'i4')])

    return note_info


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser('Generate numpy dataset')

    parser.add_argument('match_dir',
                        help='Directory containing the match files')

    parser.add_argument('--outdir',
                        help='Directory for storing the generated data',
                        default=None)

    args = parser.parse_args()

    # Default output directory
    if args.outdir is None:
        args.outdir = os.path.join(os.path.dirname(args.match_dir), 'mozart')

    # Generate if it doesn't exist
    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    # Get all match files
    all_fns = glob.glob(os.path.join(args.match_dir, '*.match'))

    for fn in all_fns:
        # Name of the piece
        name = os.path.basename(fn).replace('.match', '')
        # Generate array
        note_info = get_score_from_match(fn)

        # Out path
        out_fn = os.path.join(args.outdir, name + '.pyc.bz')
        print 'Exporting piece to {0}'.format(out_fn)
        # Export file
        save_pyc_bz(note_info, out_fn)
        # np.savez_compressed(out_fn, note_info)
