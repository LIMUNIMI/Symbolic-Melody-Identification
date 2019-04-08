#!/usr/bin/env python

import argparse
import numpy as np

from data_handling import matchfile

from data_handling.sparse_feature_extraction import score2spr
from data_handling.sparse_datafiles import csr_to_file

def summarize_match_file(m):
    """
    Display some information from a Match file

    :param m: a MatchFile object

    """

    # print header info
    for line in m.info():
        print(u'  {0}\t{1}'.format(line.Attribute,line.Value).expandtabs(20))

    # print time sig info
    print('Time signatures:')
    for t, (n, d) in m.time_signatures:
        print(' {0}/{1} at beat {2}'.format(n, d, t))


def get_notes_from_match(m):
    notes = np.array([(sn.OnsetInBeats, sn.OffsetInBeats, sn.MidiPitch[0])
                      for sn, n in m.note_pairs], np.float)
    lowest = 30
    highest = 100
    beat_div = 8
    neighbour_beats = 2
    onset_only = True
    A, _ = score2spr(notes, onset_only, lowest, highest, beat_div, neighbour_beats)
    outfile = '/tmp/sparse.npz'
    print('saving sparse matrix to {0}'.format(outfile))
    csr_to_file(outfile, A)
    
def main():
    """
    Illustrate some functionality of the match module

    """
    parser = argparse.ArgumentParser("Get information from a Matchfile file")
    parser.add_argument("file", help="Match file")
    args = parser.parse_args()

    m = matchfile.MatchFile(args.file)

    #summarize_match_file(m)
    get_notes_from_match(m)

if __name__ == '__main__':
    main()
