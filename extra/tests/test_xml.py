#!/usr/bin/env python

import argparse
import sys

import data_handling.musicxml as mxml
import data_handling.scoreontology as ont

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

def main():
    parser = argparse.ArgumentParser("Get information from a MusicXML file")
    parser.add_argument("file", help="MusicXML file")
    args = parser.parse_args()

    # the musicxml returns a list of score parts
    structure = mxml.parse_music_xml(args.file)
    for g in structure:
        print(g.pprint())




if __name__ == '__main__':
    main()
