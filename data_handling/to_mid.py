import madmom.utils.midi as midi
import os
import glob
import numpy as np
from extra.utils.os_utils import load_pyc_bz


def data_to_midi(fn, outfile):
    # Load data
    note_info = load_pyc_bz(fn)
    onset = note_info['onset']
    pitch = note_info['pitch']
    duration = note_info['duration']

    # remove grace notes
    grace_note_idxs = np.where(duration == 0)[0]
    note_info['soprano'][grace_note_idxs] = 0

    # indices for the accompaniment
    a_ix = np.where(note_info['soprano'] != 1)[0]

    # indices for the melody
    m_ix = np.where(note_info['soprano'] == 1)[0]

    # define a MIDI velocity
    velocity = np.ones_like(note_info['pitch']) * 64

    # MIDI velocity for the melody will be higher
    velocity[m_ix] += 24
    channel = np.zeros_like(note_info['pitch'])
    channel[m_ix] += 1

    # array of notes for the melody
    m_notes = np.column_stack((onset[m_ix], pitch[m_ix], duration[m_ix],
                               velocity[m_ix], channel[m_ix]))

    # array of notes for the accompaniment
    a_notes = np.column_stack((onset[a_ix], pitch[a_ix], duration[a_ix],
                               velocity[a_ix], channel[a_ix]))

    # create MIDI tracks
    tracks = [
        midi.MIDITrack().from_notes(m_notes),
        midi.MIDITrack().from_notes(a_notes),
    ]

    # Initialize MIDI file
    mf = midi.MIDIFile(tracks=tracks, file_format=0)

    # import pdb
    # pdb.set_trace()

    # Write MIDI file
    mf.write(outfile)


if __name__ == '__main__':

    data_dir = 'data/mozart'

    midi_dir = os.path.join('data', 'mozart_midi')

    if not os.path.exists(midi_dir):
        os.mkdir(midi_dir)

    data_fns = glob.glob(os.path.join(data_dir, '*.pyc.bz'))

    for fn in data_fns:

        name = os.path.basename(fn).replace('.pyc.bz', '.mid')
        outfn = os.path.join(midi_dir, name)

        print 'Saved {0} to {1}'.format(fn, outfn)

        data_to_midi(fn, outfn)
