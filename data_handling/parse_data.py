# -*- coding: utf-8 -*
"""
Functions for parsing the dataset
"""
import numpy as np
from copy import deepcopy


from extra.utils.os_utils import load_pyc_bz, save_pyc_bz


def convert_to_midi(note_array, tracks=None, save=None):
    """
    Convert a note array to a MIDIUtil MIDIFile Object

    PARAMETERS :
        note_array : a list of lists or a 2d numpy.array
            the list of notes as returned by `load_piece`
            and `convert_from_m21`; it _MUST_ contain the
            `timesignature` and `keysignature` objects.
        tracks : a list of int
            each i-th entry is the part index of the i-th note in the
            output MIDIFile object. First part has index 0.
        save : str
            the path to which the MIDIFile object is saved.
            The file type is inferred from the extension.

    RETURNS :
        a MIDIFile object.
    """

    from midiutil.MidiFile import MIDIFile, SHARPS, FLATS, MAJOR
    from math import log

    # compute the number of tracks
    if tracks:
        n_tracks = max(tracks) + 1
    else:
        n_tracks = 1

    # the new midi file object:
    mf = MIDIFile(numTracks=n_tracks, removeDuplicates=False, deinterleave=False,
                  file_format=1, ticks_per_quarternote=32, eventtime_is_ticks=False)

    # Creating indices of notes not graces
    # Not needed if the output of load_piece was better (see TODO)
    not_grace_idx = np.argwhere(note_array['duration']).reshape(-1)

    # taking values of notes not graces
    pitches = note_array['pitch'][not_grace_idx]
    onsets = note_array['onset'][not_grace_idx]
    durations = note_array['duration'][not_grace_idx]

    # scroll the notelist
    for i in range(len(pitches)):
        # is it a solo part or not?
        if tracks:
            part = tracks[i]
        else:
            part = 0

        # this is the velocity: a bit more for the solo part
        velocity = 100 - part * 25

        # this is the new note
        mf.addNote(part, part, pitches[i],
                   onsets[i], durations[i], velocity)

    # adding a tempo
    # file_format = 1: ignore the first parameter (track index)
    mf.addTempo(0, 0, 80)

    # set instrument according to GeneralMidi:
    if n_tracks == 1:
        # Piano on channel 0
        mf.addProgramChange(0, 0, 0, 0)
    else:
        # Flute on channel 0
        mf.addProgramChange(0, 0, 0, 73)
        # Piano on channel 1
        mf.addProgramChange(1, 1, 0, 0)

    # add key and time signature
    key = note_array['keysignature']
    if key.sharps < 0:
        key = (abs(key.sharps), FLATS, MAJOR)
    else:
        key = (key.sharps, SHARPS, MAJOR)

    time = note_array['timesignature']
    try:
        clock_ticks = time.beatDuration.quarterLength
    except Exception:
        clock_ticks = 24
    time = (time.numerator, log(time.denominator, 2),
            clock_ticks)
    for i in range(n_tracks):
        mf.addKeySignature(i, 0, key[0], key[1], key[2])
        mf.addTimeSignature(i, 0, time[0], time[1], time[2])

    # saving to file
    if save:
        if not save.endswith(('.midi', '.mid')):
            save += '.mid'
        with open(save, 'wb') as f:
            mf.writeFile(f)

    return mf


def convert_to_m21(note_array, tracks=None, save=None):
    """
    Convert a note array to a music21 stream.Score object

    PARAMETERS :
        note_array : a list of lists or a 2d numpy.array
            the list of notes as returned by `load_piece`
            and `convert_from_m21`; it _MUST_ contain the
            `timesignature` and `keysignature` objects.
        tracks : a list of int
            each i-th entry is the part index of the i-th note in the
            output music21 object. First part has index 0.
        save : str
            the path to which the music21 object is saved.
            The file type is inferred from the extension.

    RETURNS :
        a music21.stream.Score object
    """

    from music21 import stream, note, pitch, layout, converter, duration, instrument, midi

    # compute the number of tracks
    if tracks:
        n_tracks = max(tracks) + 1
    else:
        n_tracks = 1

    # Create parts and stream
    parts = []
    for i in range(n_tracks):
        p = stream.Part()
        p.append(deepcopy(note_array['keysignature']))
        p.append(deepcopy(note_array['timesignature']))
        parts.append(p)

    # Creating indices of notes not graces
    not_grace_idx = np.argwhere(note_array['duration']).reshape(-1)

    # taking values of notes not graces
    pitches = note_array['pitch'][not_grace_idx]
    onsets = note_array['onset'][not_grace_idx]
    durations = note_array['duration'][not_grace_idx]

    # keep memory of the last offset for computing rests
    last_offset = [9999 for i in range(n_tracks)]

    # scroll the notelist
    for index in range(len(pitches)):
        if tracks:
            part_index = tracks[index]
        else:
            part_index = 0

        # create a rest if it is needed
        rest_duration = onsets[index] - last_offset[part_index]
        if rest_duration > 0:
            d = duration.Duration(rest_duration)
            r = note.Rest()
            r.duration = d
            parts[part_index].insert(last_offset[part_index], r)

        # create the note
        n = note.Note()
        n.pitch = pitch.Pitch()
        n.pitch.midi = pitches[index]
        n.duration = duration.Duration(durations[index])

        parts[part_index].insert(onsets[index], n)
        last_offset[part_index] = onsets[index] + durations[index]

    s = stream.Score()
    for i, p in enumerate(parts):
        p.makeVoices(inPlace=True)
        # p.makeRests(fillGaps=True, inPlace=True)
        p.makeMeasures(inPlace=True)
        # p.makeTies(inPlace=True)
        # if i == 0:
        #     p.getElementsByClass('Measure')[
        #         0].insert(0.0, instrument.Flute())
        # else:
        #     p.getElementsByClass('Measure')[
        #         0].insert(0.0, instrument.Piano())
        s.insert(0, p)

    if save:
        c = converter.Converter()
        format = c.getFormatFromFileExtension(save)
        s.write(format, fp=save)

    return s


def convert_from_m21(s):
    """
    Convert a m21 stream.Stream object to a dictionary structured as follows:
        The fields are (`pitch`, `onset`, `duration`, `soprano`).
            `pitch` is the MIDI pitch of the note.
            `onset` is the score position (in beats) of the note.
            `duration` is the duration of a note (in beats).
            `soprano` is a boolean indicating whether the current note is part
                of the main melody.

    Initial pauses are discarded, that is the first notes always have onset 0.

    """
    from music21 import note, chord, key

    def add_note(n, midi_pitch):
        pitch.append(midi_pitch)
        onset.append(round(float(n.offset), 7))
        duration.append(round(float(n.duration.quarterLength), 7))
        if n_parts > 1 and hasattr(n, 'melody'):
            soprano.append(1)
        else:
            soprano.append(0)

    pitch = []
    onset = []
    duration = []
    soprano = []

    n_parts = len(s.parts)

    # marking melody notes
    if n_parts > 1:
        for melody_note in s.parts[0].recurse().getElementsByClass(['Chord', 'Note']):
            melody_note.melody = True

    # parsing all notes
    for e in s.recurse().getElementsByClass(['Chord', 'Note']):
        if type(e) is chord.Chord:
            for n in e.pitches:
                add_note(e, n.midi)
        else:
            add_note(e, e.pitch.midi)

    keys = s.recurse().getElementsByClass('KeySignature')
    if len(keys) > 0:
        keysignature = keys[0]
    else:
        keysignature = key.KeySignature(0)

    d = {
        'pitch': np.array(pitch),
        'onset': np.array(onset),
        'duration': np.array(duration),
        'soprano': np.array(soprano),
        'timesignature': s.getTimeSignatures()[0],
        'keysignature': keysignature
    }
    return d


def load_piece(fn, save=True):
    """
    Load a piece based on extension.
    If extension is '.pyc.bz', it consider it a compressed pickle
    format containing the structered array returned by this
    function, otherwise it try to load the file with music21.

    In any non-pickled format, you should care that music21 loads the melody
    as the Part object with index 0 in the stream, otherwise, if just one part
    is provided in the music21 stream, then no ground truth will be created --
    that is, the `soprano` field in the output structured array will
    be all 0. Duration and offsets loaded from other file formats will be
    rounded to the 2nd decimal.

    Initial pauses are discarded, that is the first notes always have onset 0.

    If *save* is True, then it pickles to file the structured array (if loaded
    from non-pickled format).

    TODO : remove grace notes  from the output

    Parameters
    ----------
    fn : str
        Path to the score.

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
        If the *fn* is music file (e.g. midi, musicxml) other 2 fields are added:
        `timesignature` is the music21.meter.TimeSignature as created by
            music21.converter.parse
        `keysignature` is the music21.key.KeySignature as created by
            music21.converter.parse

        Initial pauses are discarded, that is the first notes always have onset 0.
    """
    if fn.endswith('.pyc.bz'):
        return load_pyc_bz(fn)

    else:
        import music21.converter as converter
        # loading file
        s = converter.parse(fn)

        s.write('midi', fp='temp.mid')
        d = convert_from_m21(s)

        if save:
            from pathlib import Path
            p = Path(fn)
            p = p.with_suffix('.pyc.bz')
            save_pyc_bz(d, str(p))
        return d
