#!/usr/bin/env python

import sys
import os
import numpy as np
import math

from collections import defaultdict

from data_handling.scoreontology import Divisions
from data_handling.musicxml import parse_music_xml
from data_handling.matchfile import MatchFile
from utils.container_utils import partition
from utils.data_utils import normalize

import logging

LOGGER = logging.getLogger()

def makeEquivalenceClasses(s):
         return partition(lambda k:s[k],range(s.shape[0]))
def makeMeanOverEqClasses(eqcl,s):
         return dict((k,np.mean(s[v])) for k,v in eqcl.items())


def load_piece(fn_xml, fn_match, return_all_parts = False, return_units = False):
    """
    Load a musicxml file and a match file
    """

    parts = parse_music_xml(fn_xml)
    assert len(parts) > 0

    try:
        divs = float(parts[0].timeline.points[0].get_next_of_type(Divisions, True)[0].divs)
    except:
        LOGGER.error('Cannot find divs')
        sys.exit()

    match = MatchFile(fn_match)

    print(match.info)
    mcu = float(match.info('midiClockUnits'))
    mcr = float(match.info('midiClockRate'))

    
    mdict = defaultdict(list)
    for m in match.note_pairs:
        mdict[m[0].Anchor.split('-')[0]].append(m[1])
    
    if return_units:
        return (parts if return_all_parts else parts[0],
                mdict, divs, mcu, mcr)
    else:
        return (parts if return_all_parts else parts[0],
            mdict)

if __name__ == '__main__':
    xml_fn = sys.argv[1]
    match_fn = sys.argv[2]
    score,perf,divs,mcu,mcr = load_piece(xml_fn, match_fn, return_units = True)
    
    # ----------------------------------------------
    # begin - makeIOI_perBeat
    
    # score note onsets in [beats]:
    onsets = np.array([x.start.t/divs for x in score.notes])

    # perf note onsets in [midiClockUnits]:
    perf_list = np.array([perf[n.id] for n in score.notes])
    perf_mask = np.array([len(x) > 0 for x in perf_list])
    onsets = onsets[perf_mask]

    perf_list = perf_list[perf_mask]
    ids = np.arange(len(score.notes))[perf_mask]
    ponsets = np.array([x[0].Onset for x in perf_list])
    pvelocities = np.array([x[0].Velocity for x in perf_list])

    # put first perf onset to t=0:
    ponsets = ponsets-np.min(ponsets)
    # only use onsets on beats
    beatSelection = onsets % 1 == 0
    onsets_beats = onsets[beatSelection]
    ponsets_beats = ponsets[beatSelection]

    eqcl = makeEquivalenceClasses( onsets_beats)
    onset2ponset = makeMeanOverEqClasses(eqcl, ponsets_beats)
    u_onsets = np.array(sorted(eqcl.keys()))
    u_ponsets = np.array([onset2ponset[x] for x in u_onsets])

    n_u_onsets = normalize(u_onsets.copy().reshape((-1,1)))[:,0]
    n_u_ponsets = normalize(u_ponsets.copy().reshape((-1,1)))[:,0]

    ioi_r = np.diff(n_u_ponsets)/np.diff(n_u_onsets)

    # locate invalid ioi ratio values
    valid_idx = np.logical_not(np.array(ioi_r <= 0, np.bool) +
                               np.isinf(ioi_r) +
                               np.isnan(ioi_r))

    ioi_r = ioi_r[valid_idx]
    ioi_r = np.log2(ioi_r)

    xcoords = (u_onsets[1:]+u_onsets[:-1])/2
    xcoords = xcoords[valid_idx]
    
    idx = np.argsort(xcoords)
    xcoords = xcoords[idx]
    ycoords = ioi_r[idx]
    log_ioi_of_notes = np.interp(onsets, xcoords, ycoords)

    import pylab as plt
    plt.plot(xcoords, ycoords, '.')
    plt.show()
    # ----------------------------------------------
    # end - makeIOI_perBeat
      
    # print(ids.shape, pvelocities.shape)
    pvelocities = pvelocities - np.mean(pvelocities)

    result = np.column_stack((ids, pvelocities, log_ioi_of_notes))
    np.savetxt(sys.stdout,result,fmt='%.5f')
            
    # print 'Total number of score notes:', len(log_ioi_of_notes)
