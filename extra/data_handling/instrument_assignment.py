#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

The class `Instrument` represents an instrument, to be associated to a
unit of musical information, such as a MIDI track, or a staff in a
musical score. The class has both a name and a canonical name. Its
name is the textual representation of the instrument that originates
from the data. In case of reading a MIDI file, the name of the
instrument may be read from a InstrumentName or TrackName meta event
in the MIDI track. In case of analysing a musical score, it is
typically the instrument name as it is written in front of the staff
(note that in case of automatic OMR, the name may contain OCR
errors). In other words, the name of an `Instrument` is a free form
string, it may be an the instrument name in any language, or an
abreviated name.

The canonical name on the other hand is the English name for the
instrument, and is not free form, but rather one of a list of known
instrument names.

The mapping of free form instrument names to canonical instrument
names is done by the class `InstrumentMapper`. This mapper contains a
list of InstrumentName instances (`instruments`), whose purpose it is
to quantify how similar an arbitrary free form string is to the
canonical instrument name it represents. It does so by assigning a
distance value to a limited number of (user defined) normalized
strings (normalized means converted to lower case, and white-space
characters removed). If the normalized free form string starts with
any of the defined strings, the associated distance value is returned,
other wise it returns a distance of infinity, that represents no match
is possible. When an Instrument class is instantiated with a freeform
string, the Instrument class uses an InstrumentMapper instance to
guess the most likely canonical instrument name for that freeform
string.

The `InstrumentMapper` also defines a fixed mapping from canonical
instrument names to MIDI channels and patches, so that musical
information associated to an instrument can exported to MIDI without
manual intervention to set the correct MIDI patch for the
instrument. This is done through a dictionary `channel_patch_pairs`,
that maps canonical instrument names to a pair of MIDI channel, and
MIDI patch.

Finally, the `Instrument` class defines a member `rank`. The rank of an
instrument is a number that may be associated to the instrument, in
case there are multiple instruments of the same class. If during OMR a
staff with the name "Violin I,II" is encountered, two Instrument
instances will be created, both with canonical name "Violin", but one
with rank 1, and the other with rank 2.

"""


import numpy as np
import re
import logging

LOGGER = logging.getLogger(__name__)


# Note: the canonical names given here as strings have to be inserted
# into the ScoreDB's "Instrument" table exactly as written here!
PIANO = 'Piano'
HARPSICHORD = 'Harpsichord'  # ger.: Cembalo
CELESTA = 'Celesta'
ORGAN = 'Organ'
FLUTE = 'Flute'
PICCOLO = 'Piccolo'
OBOE = 'Oboe'
ENGLISHHORN = 'English Horn'
HECKELPHONE = 'Heckelphone'  # ger.: Heckelphon
CLARINET = 'Clarinet'
BASSCLARINET = 'Bass Clarinet'
BASSOON = 'Bassoon'
CONTRABASSOON = 'Contrabassoon'
SOPRANOSAXOPHONE = 'Soprano Saxophone'
ALTOSAXOPHONE = 'Alto Saxophone'
TENORSAXOPHONE = 'Tenor Saxophone'
BARITONESAXOPHONE = 'Baritone Saxophone'
FRENCHHORN = 'French Horn'
WAGNER_TUBA = 'Wagner Tuba'  # Wagner Tuba, special kind of horn. It is NOT a tenor horn! # TO DO: put in Instrument table if necessary
TRUMPET = 'Trumpet'
CORNET = 'Cornet'
TROMBONE = 'Trombone'
ALTOTROMBONE = 'Alto Trombone'
TENORTROMBONE = 'Tenor Trombone'
BASSTROMBONE = 'Bass Trombone'
TUBA = 'Tuba'  # this is actually the basstuba
CONTRABASSTUBA = 'Contrabass Tuba'  # TO DO: put in Instrument table if necessary
OPHICLEIDE = 'Ophicleide'  # predecessor to the Tuba
GLOCKENSPIEL = 'Glockenspiel'
TIMPANI = 'Timpani'
HARP = 'Harp'
VIOLIN = 'Violin'
VIOLA = 'Viola'
CELLO = 'Cello'
CONTRABASS = 'Contrabass'
SLEIGHBELLS = 'Sleigh Bells'  # aka. jingle bells; ger.: Schellen
TAMTAM = 'Tam-Tam'
CYMBALS = 'Cymbals'  # ger.: Becken
TRIANGLE = 'Triangle'
BASSDRUM = 'Bass Drum'    # ger.: große Trommel
SNAREDRUM = 'Snare Drum'  # kleine Trommel
SOPRANO_VOICE = 'Soprano Voice'
MEZZOSOPRANO_VOICE = 'Mezzosoprano Voice'
ALTO_VOICE = 'Alto Voice'
CONTRAALTO_VOICE = 'Contraalto Voice'  # rare?!
TENOR_VOICE = 'Tenor Voice'
BARITONE_VOICE = 'Baritone Voice'
BASS_VOICE = 'Bass Voice'


NON_ALPHA_NUM_PAT = re.compile(ur'\W', re.UNICODE)
# pattern that matches any leading numbers from instrument names (such
# as in '2 Violoncelli).
LEADING_NUMBERS_PAT = re.compile(ur'^([0-9\s]*)', re.UNICODE)


def normalize_string(s):
    """

    takes a string and returns it as an all lowercase string?

    Parameters
    ----------
    s : str
        the string to be normalized.

    Returns
    -------
    str
        the normalized string, i.e. all lowercase letters?
    """
    # return s.lower().replace(' ', '')
    return LEADING_NUMBERS_PAT.sub(ur'', NON_ALPHA_NUM_PAT.sub(ur'', s.lower()))


class InstrumentNameMapper(object):
    """
    """

    class InstrumentName(object):
        """
        This internal class is used to link "canonical" (english)
        instrument names to versions of the instrument names in other
        languages, and also to degraded instrument names, such as
        abbreviations, possibly with OCR errors. Each of the degraded
        name variants can gets assigned a (manually chosen) distance,
        used for ranking candidate canonical instrument names for a
        particular freeform instrument name string.

        Parameters
        ----------
        name : str
            the instrument's name in english, such as 'Piano',
            'Clarinet', 'French Horn', 'Contrabassoon', etc.

        alt_dist : dictionary
            this is a dictionary of pairs of alternative instrument
            names (and/or abbreviations of such) as expected to occur
            throughout scores, plus (manually) assigned distances
            to the "original" name that is given by `name`.
            Example: {'piano': 0} as e.g. the alternative name / distance
            pair to the "canonical" name 'Piano'.
        """

        def __init__(self, name, alt_dist):
            self.name = name
            self.alt_dist = alt_dist

        def dist(self, s):
            x = normalize_string(s)
            dist = np.inf
            for l, v in self.alt_dist.items():
                if x.startswith(l):
                    dist = min(dist, v)
            return dist

    def __init__(self):
        # in this place all possible OCR errors are reactively
        # coerced to the right instruments...
        # set up a list of InstrumentName objects that contain dicts
        # of alternative names and their respective distances to the
        # full "canonical" name
        self.instruments = [
            self.InstrumentName(PIANO, {u'piano': 0}),
            self.InstrumentName(HARPSICHORD, {u'harpsichord': 0,
                                              u'cembalo': 0}),
            self.InstrumentName(CELESTA, {u'celesta': 0,
                                          u'cel': 1}),
            self.InstrumentName(ORGAN, {u'organ': 0,
                                        u'orgel': 0}),
            self.InstrumentName(FLUTE, {u'flute': 0,
                                        u'flûte': 0,  # french
                                        u'flöte': 0,
                                        u'grfl': 0,
                                        u'großefl': 0,
                                        u'grossefl': 0,
                                        u'flaut': 1,
                                        u'fl': 2,
                                        u'f1': 2}),
            self.InstrumentName(PICCOLO, {u'picc': 0,
                                          u'flautopiccolo': 0,
                                          u'flautipiccoli': 0,
                                          u'petiteflute': 0,  # french
                                          u'petiteflûte': 0,
                                          u'klfl': 0,
                                          u'kleinefl': 0}),
            self.InstrumentName(OBOE, {u'oboe': 0,
                                       u'oboi': 0,
                                       u'hautbois': 0,  # french
                                       u'ob': 2,
                                       u'hob': 3}),
            self.InstrumentName(ENGLISHHORN, {u'englishhorn': 0,
                                              u'coranglais': 0,
                                              u'cornoingles': 0,
                                              u'engl': 1}),
            self.InstrumentName(HECKELPHONE, {u'heckelphon': 0,
                                              u'heck': 1}),
            self.InstrumentName(CLARINET, {u'clar': 1,
                                           u'klar': 1,
                                           u'kla': 1,      # check if this works
                                           u'clarinet': 0,
                                           u'klarinet': 0,
                                           u'petiteclarinet': 1,  # let's see if that works (Eb clarinet)
                                           u'cl': 2,
                                           u'c1': 2}),
            self.InstrumentName(BASSCLARINET, {u'bassclar': 0,
                                               u'bassklar': 0,
                                               u'baßklar': 0,
                                               u'bcl': 0,
                                               u'bkl': 0,
                                               u'clarinettobasso': 0}),
            self.InstrumentName(BASSOON, {u'fag': 0,
                                          u'bassoon': 0,
                                          u'basson': 0,  # french
                                          u'bs': 2}),
            self.InstrumentName(CONTRABASSOON, {u'contrafag': 0,
                                                u'kontrafag': 0,
                                                u'ctrfg': 0,
                                                u'ctrfag': 0,
                                                u'cfag': 0,
                                                u'kfag': 0,
                                                u'contrabassoon': 0,
                                                u'doublebassoon': 0}),  # problems with double bass?
            self.InstrumentName(SOPRANOSAXOPHONE, {u'sopranosax': 0,
                                                   u'sopransax': 0}),
            self.InstrumentName(ALTOSAXOPHONE, {u'altosax': 0,
                                                u'altsax': 0}),
            self.InstrumentName(TENORSAXOPHONE, {u'tenorsax': 0}),
            self.InstrumentName(BARITONESAXOPHONE, {u'baritonesax': 0,
                                                    u'baritonsax': 0}),
            self.InstrumentName(FRENCHHORN, {u'frenchhorn': 0,
                                             u'horn': 0,
                                             u'hörn': 0,
                                             u'hn': 1,
                                             u'cor': 1,
                                             u'como': 2,
                                             u'c0r': 2}),
            self.InstrumentName(WAGNER_TUBA, {u'wagnertub': 0,
                                              u'tenortub': 0,       # c.f. e.g. Alpensinfonie
                                              u'wagnerhorn': 0,     # not very common?
                                              u'bayreuthtub': 0}),  # not very common?
            self.InstrumentName(TRUMPET, {u'trumpet': 0,
                                          u'clarin': 0,  # clarino, clarini
                                          u'tromba': 0,
                                          u'trombe': 0,
                                          u'trompet': 0,
                                          u'trp': 1,
                                          u'tpt': 2,
                                          u'clno': 2}),
            self.InstrumentName(CORNET, {u'kornett': 0,
                                         u'cornet': 0}),
            self.InstrumentName(TROMBONE, {u'trombo': 1,
                                           u'posaun': 0,
                                           u'tbn': 1,
                                           u'tr': 2}),
            self.InstrumentName(ALTOTROMBONE, {u'trombonealt': 0,
                                               u'altpos': 0,
                                               u'atr': 0,
                                               u'atbn': 0,
                                               u'alttr': 0,
                                               u'altotr': 0}),
            self.InstrumentName(TENORTROMBONE, {u'trombonetenor': 0,
                                                u'tenorpos': 0,
                                                u'ttr': 0,
                                                u'ttbn': 0,
                                                u'tentr': 0,
                                                u'tenortr': 0}),
            self.InstrumentName(BASSTROMBONE, {u'trombonebass': 0,
                                               u'basspos': 0,
                                               u'basstbn': 0,
                                               u'tbnbasso': 0,
                                               u'tbnbass': 0,
                                               u'basstromb': 0}),
            self.InstrumentName(TUBA, {u'tuba': 0,
                                       u'tb': 2,
                                       u'basstuba': 1,
                                       u'baßtuba': 1}),
            self.InstrumentName(CONTRABASSTUBA, {u'contrabasstub': 0,
                                                 u'kontrabasstub': 0,
                                                 u'kontrabaßtuba': 0,
                                                 u'tubacontrebass': 0,
                                                 u'tubacontrabass': 0}),
            self.InstrumentName(OPHICLEIDE, {u'ophicleide': 0,
                                             u'ophicléide': 0,  # french
                                             u'ophikleide': 0}),
            self.InstrumentName(GLOCKENSPIEL, {u'glockenspiel': 0,
                                               u'bells': 0,
                                               u'glspl': 0}),
            self.InstrumentName(TIMPANI, {u'timp': 0,
                                          u'timbale': 0,
                                          u'pauk': 0}),
            self.InstrumentName(HARP, {u'harp': 0,
                                       u'harfe': 0,
                                       u'hfe': 1}),
            self.InstrumentName(VIOLIN, {u'violin': 0,
                                         u'violon': 1,  # french, balance with violoncell!
                                         u'geige': 0,   # german
                                         u'gg': 1,      # german, abbrev. "Gg." for "Geige"
                                         u'vln': 2,
                                         u'vni': 2,
                                         u'soloviolin': 2}),
            self.InstrumentName(VIOLA, {u'viola': 0,
                                        u'viole': 0,
                                        u'bratsche': 0,
                                        u'br': 1,
                                        u'vla': 2,
                                        u'vle': 2}),
            self.InstrumentName(CELLO, {u'violoncell': 0,
                                        u'voiloncello': 2,
                                        u'cello': 0,
                                        u'vlc': 2,
                                        u'vc': 2}),
            self.InstrumentName(CONTRABASS, {u'contrabass': 0,
                                             u'contrebass': 0,  # french
                                             u'doublebass': 0,
                                             u'kontrabass': 0,
                                             u'kontraba': 1,
                                             u'bassi': 0,
                                             u'basso': 1,
                                             u'cb': 0,
                                             u'db': 1}),
            self.InstrumentName(SLEIGHBELLS, {u'sleighbell': 0,
                                              u'jinglebell': 0,
                                              u'schelle': 0,
                                              u'sche': 2,
                                              u'sch': 3}),  # dangerous?
            self.InstrumentName(TAMTAM, {u'tamtam': 0,
                                         u'tam-tam': 0,  # necessary?
                                         u'tam': 2}),
            self.InstrumentName(CYMBALS, {u'cymb': 0,
                                          u'beck': 0,
                                          u'cin': 2,  # cinelli, etc.
                                          u'bck': 2}),
            self.InstrumentName(TRIANGLE, {u'triang': 0,
                                           u'tri': 1,
                                           u'trgl': 2}),
            self.InstrumentName(BASSDRUM, {u'bassdr': 0,
                                           u'großetr': 0,
                                           u'grossetr': 0,
                                           u'grantamb': 0,
                                           u'grtamb': 0,
                                           u'tamburogr': 0,
                                           u'grcaisse': 0,
                                           u'grossecaisse': 0,
                                           u'grancassa': 0,
                                           u'grcassa': 0,
                                           u'grtr': 3}),
            self.InstrumentName(SNAREDRUM, {u'tambour': 0,  # french
                                            u'snaredrum': 0}),
            self.InstrumentName(SOPRANO_VOICE, {u'sopran': 0}),
            self.InstrumentName(MEZZOSOPRANO_VOICE, {u'mezzosopran': 0}),
            self.InstrumentName(ALTO_VOICE, {u'alt': 0}),
            self.InstrumentName(CONTRAALTO_VOICE, {u'contraalt': 0}),
            self.InstrumentName(TENOR_VOICE, {u'tenor': 0}),
            self.InstrumentName(BARITONE_VOICE, {u'bariton': 0}),
            self.InstrumentName(BASS_VOICE, {u'bass': 1,
                                             u'bassi': 3})  # check with contrabass
            ]

        # channel numbers are in the range 1-16,
        # program (patch) numbers are in the range 0-127.
        self.channel_patch_pairs = {PIANO: (1, 0),
                                    HARPSICHORD: (1, 6),
                                    CELESTA: (1, 8),
                                    ORGAN: (1, 19),
                                    FLUTE: (1, 73),
                                    PICCOLO: (1, 72),  # has own program number
                                    OBOE: (6, 68),
                                    ENGLISHHORN: (5, 69),
                                    HECKELPHONE: (6, 68),  # mapped to oboe
                                    CLARINET: (3, 71),
                                    BASSCLARINET: (3, 71),  # mapped to clarinet
                                    BASSOON: (2, 70),
                                    CONTRABASSOON: (2, 70),  # mapped to basson
                                    SOPRANOSAXOPHONE: (4, 64),
                                    ALTOSAXOPHONE: (4, 65),
                                    TENORSAXOPHONE: (4, 66),
                                    BARITONESAXOPHONE: (4, 67),
                                    FRENCHHORN: (5, 60),
                                    WAGNER_TUBA: (5, 60),  # mapped to french horn
                                    TRUMPET: (13, 56),
                                    CORNET: (13, 56),  # mapped to trumpet
                                    TROMBONE: (7, 57),
                                    ALTOTROMBONE: (7, 57),   # mapped to trombone
                                    TENORTROMBONE: (7, 57),  # mapped to trombone
                                    BASSTROMBONE: (7, 57),   # mapped to trombone
                                    TUBA: (14, 58),
                                    CONTRABASSTUBA: (14, 58),  # mapped to tuba
                                    OPHICLEIDE: (14, 58),      # mapped to tuba
                                    GLOCKENSPIEL: (13, 9),
                                    TIMPANI: (12, 47),
                                    HARP: (14, 46),
                                    VIOLIN: (9, 48),
                                    VIOLA: (9, 48),
                                    CELLO: (11, 48),
                                    CONTRABASS: (4, 48),
                                    SLEIGHBELLS: (10, 48),  # (channel 10 reserved for percussion, 48 = orchestral percussion kit)
                                    TAMTAM: (10, 48),
                                    CYMBALS: (10, 48),
                                    TRIANGLE: (10, 48),
                                    BASSDRUM: (10, 48),
                                    SNAREDRUM: (10, 48),
                                    SOPRANO_VOICE: (16, 53),  # all voices are mapped to patch 'Voice Oohs'
                                    MEZZOSOPRANO_VOICE: (16, 53),
                                    ALTO_VOICE: (16, 53),
                                    CONTRAALTO_VOICE: (16, 53),
                                    TENOR_VOICE: (16, 53),
                                    BARITONE_VOICE: (16, 53),
                                    BASS_VOICE: (16, 53),
                                    }

    def map_instrument(self, s):
        """
        function that tries to map the given string to a canonical
        instrument name.

        Parameters
        ----------
        s : str
            the freeform name (identifier, label) of the instrument.

        Returns
        -------
        str OR None
            the canonical instrument name if one was found.
            Else, None is returened.
        """

        dists = np.zeros(len(self.instruments))
        for i, instr in enumerate(self.instruments):
            dists[i] = instr.dist(s)
        i = np.argmin(dists)
        if dists[i] < np.inf:    # smaller than infinity?
            return self.instruments[i].name
        else:
            # LOGGER.warning(u'No canonical name could be found for "{0}"'.format(s))
            return None


class UnkownTranspositionException(BaseException):
    pass


class Instrument(object):
    """
    A class that represents an individual instrument.

    It is instantiated from a MIDI track name or an OCR-extracted staff
    name. The name is coerced (if possible) to a canonical instrument
    name, and if applicable, the "rank" of the instrument (e.g. Viola
    1, Viola 2) is determined.

    Parameters
    ----------
    name : str
        the instrument's name (identifier, label) to be coerced to
        a "canonical" name

    Attributes
    ----------
    name : str

    canonical_name : str OR None

    transposition : integer OR None
        the transposition of the instrument in +/- semitones.

    rank : number OR None
        the rank here means e.g. Viola 1 vs Viola 2.

    channel : number
        the MIDI channel number assigned to the instrument. Should be
        in the range of 1-16. Note that channel 10 is reserved for
        unpitched percussion instruments in the General Midi standard.

    patch : number
        the MIDI patch (program) number assigned to the instrument.
        This should be the General MIDI (GM) number for that instrument,
        usually in the range of 0-127.
        Note that unpitched percussion instruments (should be set to chn 10)
        will typically receive a patch number that is used to select a
        drum set (e.g. patch 48 for orchestral percussion set).
    """

    im = InstrumentNameMapper()
    rank_pat = re.compile('\s(?:(3|III|iii)|(2|ii|II)|([iI1]))($|\W)')

    def __init__(self, name):
        self.name = name
        self.canonical_name = self.im.map_instrument(normalize_string(name))
        # self.transposition = estimate_transposition(self.canonical_name, self.name)
        self.rank = self.estimate_rank()
        self.channel, self.patch = self.im.channel_patch_pairs.get(self.canonical_name, (None, None))

    def __unicode__(self):
        return u'{0} ({1}/{2}/{3})'.format(self.name, self.canonical_name,
                                           self.rank, self.transposition)

    def __str__(self):
        return self.__unicode__().encode('utf8')

    def estimate_rank(self):
        """
        estimate the rank of the instrument, e.g. first Violin vs second
        Violin, etc.

        Returns
        -------
        number OR None
        """

        # use the reg ex given above to estimate the rank
        m = self.rank_pat.search(self.name)
        if m:
            if m.group(1):
                return 3
            elif m.group(2):
                return 2
            elif m.group(3):
                return 1
        else:
            return None

    @property  # check whether this breaks anything
    def transposition(self):
        return estimate_transposition(self.canonical_name, self.name)


def decompose_name(s):
    """
    If `s` is a conjunction of two instrument names (e.g. "Violoncello
    e Contrabasso"), return both parts. Furthermore, if `s` contains a
    conjunction, return both parts, e.g. return ("Viola 1", "Viola 2")
    for "Viola 1,2".

    Parameters
    ----------
    s : str

    Returns
    -------
    list of str
    """

    s_lower = s.lower()
    conjunction_pat = re.compile('(?:\+|&| e | i )')
    parts = conjunction_pat.split(s_lower)

    if len(parts) > 1:
        return parts

    onetwo_pat = re.compile(ur'(.*)(?:([iI1])[&,-\u2014]\s?(2|ii|II))(.*)', re.UNICODE)
    m = onetwo_pat.search(s)
    if m:
        pre, g1, g2, post = m.groups()
        if not pre.endswith(' '):
            pre = pre + ' '
        return [u''.join((pre, g1, post)),
                u''.join((pre, g2, post))]
    else:
        return [s]


def assign_instruments(instr_sc_channel_map, extracted_staff_names):
    """
    This function is used for mapping OMR/OCR information about staffs
    to a list of instruments. This function should be elsewhere, not
    in instrument_assignment.py ...

    Parameters
    ----------
    instr_sc_channel_map :

    extracted_staff_names :

    Returns
    -------
    score_channels : list

    """

    sc_instruments = [(Instrument(k), v) for k, v in instr_sc_channel_map.items()]

    sc_instruments_map = dict(((instr.canonical_name, instr.rank), (instr, sc))
                              for instr, sc in sc_instruments)

    for k, v in sc_instruments_map.items():
        print(k, v[0].name)
    print('')
    for k in extracted_staff_names:
        print(k)
    print('')

    score_channels = []

    for k in extracted_staff_names:
        names = decompose_name(k)
        instruments = [Instrument(n) for n in names]
        print(u'{0}:'.format(k))
        esc = []
        #print(k, names, instruments)
        for instr in instruments:
            #print((instr.canonical_name, instr.rank))
            mapped_instr, score_channel = sc_instruments_map.get((instr.canonical_name, instr.rank), (None, None))
            if mapped_instr:
                print(u'\t{0}/{1} ({2})'.format(mapped_instr.canonical_name, mapped_instr.rank, score_channel))
                esc.append(score_channel)
            else:
                found_cn = False
                if instr.rank is None:
                    # it may be that there are rank 1,2 instruments, and this staff refers to both implicitly
                    for (cn, r), (sc_i, sc_channel) in sc_instruments_map.items():
                        if cn == instr.canonical_name:
                            print(u'\t{0}/{1} ({2})'.format(cn, r, sc_channel))
                            found_cn = True
                            esc.append(sc_channel)
                if instr.rank is not None or not found_cn:
                    print('ERROR', k, instr.name, instr.canonical_name, instr.rank)
        score_channels.append(esc)
        print('')

    print(score_channels)

    return score_channels

# TRANSPOSITIONS contains the possible transpositions of an instrument
# type, and regular expression patterns that trigger the
# transpositions in a freeform instrument name (e.g. "Horn in
# F"). TRANSPOSITIONS is a dictionary where keys are canonical
# instrument names (defined as global variables earlier in this
# file). The values are either:
#
# 1. An integer, representing the transposition in semitones for this
#    instrument (without inspecting the freeform name)
#
# 2. A tuple of pairs, where the first element of each pair is a
#    regular expression, and the second element is either of form
#    1. or of form 2.
#
# The nesting of the regular expressions defines a nested structure of
# `if-elif` clauses

TRANSPOSITIONS = {
    CLARINET:  # is this ignored for e.g. a C clarinet (that doesn't need transposition)?
    (
        (re.compile(u'(sul|in|en) (Sib|B)', re.UNICODE), -2),
        (re.compile(u'(sul|in|en) (A|La)', re.UNICODE), -3),
        (re.compile(u'(sul|in|en) (Es|Mib)', re.UNICODE), +3),
        (re.compile(u'(sul|in|en) (D|Re)', re.UNICODE), +2)
    ),
    TRUMPET:
    (
        (re.compile(u'(sul|in|en) (Sib|B)', re.UNICODE), -2),
        (re.compile(u'(sul|in|en) (D|Re)', re.UNICODE), +2),
        (re.compile(u'(sul|in|en) (Fa|F)', re.UNICODE), +5),
        (re.compile(u'(sul|in|en) (Es|Mib)', re.UNICODE), +3)
    ),
    CORNET:
    (   # This here is introduced first time for Berlioz symphonie fantastique
        # which calls for "Cornet a Pistons", which seems to be an ancestor to
        # the modern Cornet. The score calls for a "Cornet a Pistons en Sol",
        # therefore a transpostion for key of G is given here, derived from
        # how a trumpet in G is notated.
        (re.compile(u'(sul|in|en) (Sib|B)', re.UNICODE), -2),
        (re.compile(u'(sul|in|en) (G|Sol)', re.UNICODE), +7)   # like a trumpet in G would be. Correct?
    ),
    FRENCHHORN:
    (
        (re.compile(u'(sul|in|en) (Fa|F)', re.UNICODE), -7),
        (re.compile(u'(sul|in|en) (Es|Mib)', re.UNICODE), -9),
        (re.compile(u'(sul|in|en) (E|Mi)', re.UNICODE), -8),     # e.g. symphonie fantastique
        (re.compile(u'(sul|in|en) (D|Re)', re.UNICODE), -10),
        (re.compile(u'(sul|in|en) (Sib|B)', re.UNICODE),
            (
                (re.compile(u'basso|tief|grave', re.UNICODE), -14),
                (re.compile(u'alto|hoch|haut', re.UNICODE), -2),
                # this is  a fallback that should
                # give at least the correct key.
                (re.compile(u'', re.UNICODE), -2)
            )
        ),
    ),
    WAGNER_TUBA:
    (   # The Tenor Wagner Tuba in B may in modern notation be written
        # like a French Horn and would thus have to be transposed by
        # -7 semitones instead of the -2. This would have to be checked
        # and some workaround would have to be used.
        # TO DO: for the future: make it possible to override the
        # transposition defined here from the outside when the specific
        # piece/score requieres it?
        (re.compile(u'(sul|in|en) (Sib|B)', re.UNICODE), -2),  # Tenor Wagner Tuba
        (re.compile(u'(sul|in|en) (Fa|F)', re.UNICODE), -7)    # Bass Wagner Tuba
    ),
    BASSCLARINET:
    (
        (re.compile(u'(sul|in|en) (Sib|B)', re.UNICODE), -14),
        (re.compile(u'(sul|in|en) (La|A)', re.UNICODE), -15)
    ),
    PICCOLO: +12,
    # (
    #     (re.compile(u'(sul|in|en) (Do|C)', re.UNICODE), +12),
    #     (re.compile(u'(sul|in|en) (Des|Reb)', re.UNICODE), +13)  # may occur
    # )
    ENGLISHHORN: -7,
    CONTRABASS: -12,
    CONTRABASSOON: -12
}


def _est_transp_recursive(transp, name):
    """
    Internal function used by `estimate_transposition`

    """

    if isinstance(transp, int):
        return transp
    else:
        try:
            for pattern, result in transp:
                if pattern.search(name) is not None:
                    # print('matching name to', name, pattern.pattern)
                    est = _est_transp_recursive(result, name)
                    return est
            return None
        except:
            Exception(('Format error in TRANSPOSITIONS detected while '
                       'determining transposition for instrument "{0}"')
                      .format(name))


def estimate_transposition(canonical_name, name):
    """
    Lookup transpositions of instruments, given the instrument type
    `canonical_name` (should match one of the global variable
    instrument names), and a freeform string `name`. This function
    checks `name` for transposition information (such as "in Mi"), and
    returns a transposition (in semitones) accordingly.

    This function relies on a list of nested pairs of regular
    expressions, and transpositions, defined in a global variable
    TRANSPOSITIONS

    Returns
    -------
    integer OR None
        the transposition in +/- semitones.
    """

    instr_transp = TRANSPOSITIONS.get(canonical_name, None)
    if instr_transp is None:
        return None
    else:
        return _est_transp_recursive(instr_transp, name)

# def estimate_transposition_obsolete(canonical_name, name):
#     try:
#         if canonical_name == FRENCHHORN:
#             if 'Fa' or 'F' in name:
#                 return -7
#             elif 'Sib' or 'B' in name:
#                 if 'basso' in name:
#                     return -14
#                 elif 'alto' in name:
#                     return -2
#                 else:
#                     raise UnkownTranspositionException()
#             elif 'Es' or 'Mib' in name:
#                 return -9
#             elif 'D' or 'Re' in name:
#                 return -10
#             else:
#                 raise UnkownTranspositionException()
#         elif canonical_name == CLARINET:
#             #if 'in Sib' or 'in B' in name:
#             if b_pat.search(name):
#                 print('clar b pat matches')
#                 return -2
#             # elif 'in A' or 'A' or 'La' in name:
#             elif a_pat.search(name):
#                 print('clar a pat matches')
#                 return -3
#             elif es_pat.search(name): # 'Es' or 'Mib' in name:
#                 print('clar es pat matches')
#                 return +3
#             else:
#                 raise UnkownTranspositionException()
#         elif canonical_name == BASSCLARINET:
#             if 'Sib' or 'B' in name:
#                 return - 14
#             else:
#                 raise UnkownTranspositionException()
#         elif canonical_name == ENGLISHHORN:
#             return -7
#         elif canonical_name == TRUMPET:
#             if 'Sib' or 'B' in name:
#                 return -2
#             elif 'D' or 'Re' in name:
#                 return +2
#             elif 'Es' or 'Mib' in name:
#                 return +3
#             elif 'F' or 'Fa' in name:
#                 return +5
#             raise UnkownTranspositionException()
#         else:
#             return 0
#     except UnkownTranspositionException as e:
#         LOGGER.warning(u'{0} with unknown transposition: "{1}", assuming no transposition'.format(canonical_name, name))
#         return 0
