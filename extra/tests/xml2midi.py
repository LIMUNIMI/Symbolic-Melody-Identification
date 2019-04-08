#!/usr/bin/env python

import argparse
import sys

import lbm.extra.data_handling.musicxml as mxml
import lbm.extra.data_handling.scoreontology as ont
import lbm.extra.data_handling.midi as midi
from lbm.extra.data_handling.instrument_assignment import Instrument
 
def get_notes_from_part(part):
    """
    print the notes of a score part

    :param part: a ScorePart object

    """

    tl = part.timeline
    assert len(tl.points) > 0

    tl.lock()

    div = 4.0
    print('part', part)
    f = open('/tmp/out.txt', 'w')
    for tp in tl.points:
        notes = tp.starting_objects.get(ont.Note, [])
        tp_divs = tp.starting_objects.get(ont.Divisions, [])

        if len(tp_divs) > 0:
            div = float(tp_divs[0].divs)

        for n in notes:
            print('{0} {1} {2}'.format(n.start.t/div, n.end.t/div, n.midi_pitch))
            f.write('{0} {1} {2}\n'.format(n.start.t/div, n.end.t/div, n.midi_pitch))
    f.close()

def write_score_part(g, m, instrument = None):
    g.expand_grace_notes()

    tdiv_midi = m.header.time_division
    events = []
    channel = 1

    if instrument != None:
        instr = Instrument(instrument)
    else:
        instr = Instrument(g.part_name)

    if instr != None:
        channel = instr.channel
        events.append(midi.TrackNameEvent(0, g.part_name))
        events.append(midi.PatchChangeEvent(0, channel, instr.patch))
    tdiv_xml = 1
    c_events = m.tracks[0].events
    print(instr)
    print(channel)
    sharps = 'FCGDAEB'
    flats = 'BEADGCF'

    alts = ''
    sign = 0

    transp = 0
    for tp in g.timeline.points:

        t = tdiv_midi*float(tp.t)/tdiv_xml
        # assert t % 1 == 0
        t = int(t)

        tss =  tp.starting_objects.get(ont.TimeSignature, [])
        if len(tss) > 0:
            # print(t, 'TimeSig', tss[0].beats, tss[0].beat_type)
            c_events.append(midi.TimeSigEvent(t, tss[0].beats, tss[0].beat_type))

        trs =  tp.starting_objects.get(ont.Transposition, [])
        if len(trs) > 0:
            if trs[0].chromatic:
                transp = trs[0].chromatic
                print('trs',transp)
            else:
                transp = 0

        kss =  tp.starting_objects.get(ont.KeySignature, [])
        if len(kss) > 0:
            # print(t, 'KeySig', kss[0].fifths, kss[0].mode)
            if kss[0].fifths < 0:
                alts = flats[:-kss[0].fifths]
                sign = -1
            elif kss[0].fifths < 0:
                alts = sharps[:kss[0].fifths]
                sign = +1
            else:
                alts = ''
                sign = 0

            #c_events.append(midi.TimeSigEvent(t, tss[0].beats, tss[0].beat_type))

        offs =  tp.ending_objects.get(ont.Note, [])
        for n in offs:
            # print(n.step, n.octave, n.alter)
            events.append(midi.NoteOffEvent(t, channel, n.midi_pitch + transp, 60))

        ons =  tp.starting_objects.get(ont.Note, [])
        for n in ons:
            events.append(midi.NoteOnEvent(t, channel, n.midi_pitch + transp, 60))

        tdiv =  tp.starting_objects.get(ont.Divisions, [])
        if len(tdiv) > 0:
            tdiv_xml = tdiv[0].divs

    m.add_track(midi.MidiTrack(events))

def write_tracks(structure, m, instrument = None):
    for g in structure:
        if isinstance(g, ont.ScorePart):
            write_score_part(g, m, instrument)
        elif isinstance(g, ont.PartGroup):
            write_tracks(g.constituents, m)
    return m

def main():
    parser = argparse.ArgumentParser("Get information from a MusicXML file")
    parser.add_argument("file", help="MusicXML file")
    parser.add_argument("outfile", help="MIDI output filename")
    parser.add_argument("--instrument", help="An instrument name")
    args = parser.parse_args()

    # the musicxml returns a list of score parts
    structure = mxml.parse_music_xml(args.file)
    m = midi.MidiFile()
    tdiv_midi = 1024
    m.header = midi.MidiHeader(1, time_division = tdiv_midi)
    m.add_track(midi.MidiTrack())
    write_tracks(structure.constituents, m, args.instrument)
    #for g in structure:
        #print(g.pprint())
    m.write_file(args.outfile)




if __name__ == '__main__':
    main()
