#!/usr/bin/env python

import argparse
import pylab as plt

from data_handling import wormfile

def plot_worm_data(w):
    """
    Plot the tempo, loudness, and structural annotations of a worm
    file against time.
    
    :param w: a WormFile object

    """

    ax1 = plt.subplot(313)
    plt.plot(w.data[:,0],w.data[:,3],'|')
    plt.ylabel('Structure')
    plt.xlabel('Time (seconds)')

    ax2 = plt.subplot(312, sharex=ax1)
    plt.plot(w.data[:,0],w.data[:,2])
    plt.ylabel('Loudness (Sone)')
    plt.setp( ax2.get_xticklabels(), visible=False)

    ax3 = plt.subplot(311, sharex=ax1)
    plt.plot(w.data[:,0],w.data[:,1])
    plt.ylabel('Tempo (BPM)')
    plt.setp( ax3.get_xticklabels(), visible=False)

    plt.title(u'{0} by {1} ({2})'.format(w.header['Piece'],
                                         w.header['Performer'],
                                         w.header['YearOfRecording']))

    plt.show()


def summarize_worm_file(w):
    """
    Print the header information of a worm file.
    
    :param w: a WormFile object

    """

    print("Worm header information:")
    for k,v in w.header.items():
        print(u'  {0}:\t{1}'.format(k, v).expandtabs(20))


def main():
    """
    Illustrate some functionality of the worm module

    """
    parser = argparse.ArgumentParser("Get information from a Worm file")
    parser.add_argument("file", help="Worm file")
    parser.add_argument("-p", "--plot", action = 'store_true',
                        help = 'plot Worm data')
    args = parser.parse_args()

    w = wormfile.WormFile(args.file)

    summarize_worm_file(w)
    w.write_file('/tmp/tst.worm')
    if args.plot:
        plot_worm_data(w)


if __name__ == '__main__':
    main()
