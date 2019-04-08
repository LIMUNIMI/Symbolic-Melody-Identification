#!/usr/bin/env python2
from nn_models import rnn
from nn_models import cnn
from melody_extractor import graph_tools
from melody_extractor import settings
from melody_extractor import misc_tools
from utils import pianoroll_utils
from data_handling import parse_data
import numpy as np
import json
import os
import sys
from argparse import RawTextHelpFormatter
import argparse
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


try:
    import cPickle as pickle
except ImportError:
    import pickle

DEFAULT_MODEL = '12345model.pkl'


class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)


def check_args():
    parser = MyParser(
        description='Takes in input a model and a symbolic music file and\n\
outputs a new symbolic music file containing solo part and accompaniment\n\
separation. At least one and no more than one option must be provided\n\
between `--extract`, `--train`, `--crossvalidation`, `--hyperopt`, `--rebuild`.\n\
If more than one is provided the first one in the previous list will be used.',
        prog=sys.argv[0], formatter_class=RawTextHelpFormatter)

    parser.add_argument('--extract', metavar=('INPUT', 'OUTPUT'),
                        type=str, nargs=2, default=[],
                        help='Extract a solo part from INPUT and create a new\n\
    OUTPUT file containing the accompaniment and the solo\n\
    part in two separeted tracks (if midi) or parts (if musicxml)\n\
The file types are inferred from the extensions.\n\
Supported input files: MIDI (.mid, .midi), MusicXML (.xml, .mxl),\n\
    ABC (.abc), Kern (.krn), and anything supported by music21 library\n\
    In addition, it supports the loading of pickled objects(.pyc.bz)\n\
    containing structered arrays: \n\
    * Each row corresponds to a note.\n\
    * The fields must be(`pitch`, `onset`, `duration`, `soprano`).\n\
            `pitch` is the MIDI pitch of the note.\n\
            `onset` is the score position(in beats) of the note.\n\
            `duration` is the duration of a note(in beats).\n\
            `soprano` is a boolean indicating whether the current\n\
                note is part of the main melody.\n\
Supported output files: MIDI(.mid, .midi) type 1; ".mid" extension\n\
    will be added if not provided.\n')

    parser.add_argument('--inspect-masking', metavar=('INPUT', 'N'),
                        type=str, nargs=2, default=[],
                        help='Perform inspection by using masking windows\n\
on the window number N of song INPUT.\n\
Create numpy compressed file containing the input pianoroll (`in_pianoroll`),\n\
the output pianoroll (`out_pianoroll`), the ground truth melody (`melody`),\n\
the masked saliency map (`saliency`) and the whole masked outputs (`masked`)\n')

    parser.add_argument('--inspect', metavar=('INPUT'),
                        type=str, nargs=1, default=[],
                        help='Perform standard inspection on the INPUT song.\n\
Create numpy compressed file containing the input pianoroll (`in_pianoroll`),\n\
the output pianoroll (`out_pianoroll`), the ground truth melody (`melody`)\n\
and the salience map.\n\
The file types are inferred from the extensions, as in `--extract`.\n')

    parser.add_argument('--model', metavar='PATH',
                        default=DEFAULT_MODEL,
                        # nargs=1,
                        help="A pickled object containing a `predict` method. Usually it's\n\
    a CNN or RNN object. By default the trained models provided with\n\
    the software will be used. If a custom network is provided, you\n\
    should take care that it comes with a `win_width` field, as provided\n\
    by `melody_extractor.trainer.build_?NN_model`\n")

    parser.add_argument('--install-deps', action='store_true',
                        help="Automatically install missing libraries in the user home\n")

    parser.add_argument('--check-deps', action='store_true',
                        help="Automatically check needed libraries on startup. If\n\
    `--install-deps` is not used, a command to install the missing\n\
    dependencies will be printed\n")

    parser.add_argument('--rnn', action='store_true',
                        help="Use an RNN and non-overlapping windows instead of a CNN\n\
    with overlapping windows\n")

    parser.add_argument('--time-limit', metavar='INT',
                        default=120,
                        type=int,
                        help="Break the training if the time exceeds the specified\n\
    limit in seconds (default 120 sec)")

    parser.add_argument('--epochs', metavar='INT',
                        default=15000,
                        type=int,
                        help="Set the maximum number of epochs. Default 15000.\n\
    Note that the training already uses early stop algorithm. This option is\n\
    particularly useful for RCNNs.\n")

    parser.add_argument('--mono', action='store_true',
                        help="Find a strictly monophonic solo part.\n")

    parser.add_argument('--train', metavar=('DIR', '.EXT', 'FILE'),
                        default=[], nargs=3,
                        help="Train the model on files in DIR (and subdirectories)\n\
    having extension .EXT. Write the trained model to a pickled\n\
    object in this directory. Use parameters contained in FILE.\n\
    Wite also parameters to be used with `--rebuild` option for\n\
    rebuilding the network on a different architecture\n")

    parser.add_argument('--crossvalidation', metavar=('DIR', '.EXT', 'FILE'),
                        default=[], nargs=3,
                        help="Perform a 10-fold crossvalidation on files in DIR\n\
    (and subdirectories) having extension .EXT. Write results to\n\
    files in this directory. Use parameters contained in FILE as\n\
    exported with `--hyper-opt`. At each fold, it save a pickled object\n\
    containing the kernels of the network; you can use these to rebuild\n\
    the network on a different architecture (`--rebuild` option)\n")

    parser.add_argument('--validate', metavar=('DIR', '.EXT', 'MODEL'),
                        type=str, nargs=3, default=[],
                        help='Validate the MODEL on files in DIR and sub-dir\n\
of type .EXT. Writes the results in a file in the current directory\n\
called `results.txt`. This only works with CNN.\n')

    parser.add_argument('--rebuild', metavar=('KERNELS', 'PARAMETERS', 'OUTPUT'),
                        default=[], nargs=3,
                        help="This is provided for convenience: after\n\
    having trained a model (`--train`), use the output kernels\n\
    parameters and the parameters found with `--hyper-opt` to \n\
    build a new model on a different architecture without retrain\n\
    retrain it. This is useful if you train on GPU and then want\n\
    the model exported for a CPU. Write the model in OUTPUT.\n")

    parser.add_argument('--hyper-opt', metavar=('DIR', '.EXT', 'FILE'),
                        default=[], nargs=3,
                        help="Perform hyper-parameter optimization on files in DIR\n\
    (and subdirectories) having extension .EXT. This write the best parameters\n\
    FILE at each new evaluation. If for any reason the hyper-optimization\n\
    should stop, then you should take care that `trials.hyperopt` is still\n\
    in the working directory, so that the already performed evaluations will\n\
    not be lost.\n\
    \n\
    N.B. Be careful about the output parameters because something seems to be\n\
    written wrong (maybe an hyper-opt bug?)\n")

    args = vars(parser.parse_args(None if sys.argv[1:] else ['-h']))
    return args


def check_libraries(args):
    missing = ""
    try:
        import numpy
    except ImportError:
        missing += ' numpy'

    try:
        import music21
    except ImportError:
        missing += ' music21'

    try:
        from midiutil import MIDIFile
    except ImportError:
        missing += ' MIDIUtil'

    try:
        import theano
    except ImportError:
        missing += ' theano'

    try:
        import lasagne
    except ImportError:
        missing += ' lasagne'

    try:
        import sklearn
    except ImportError:
        missing += ' sklearn'

    try:
        import matplotlib
    except ImportError:
        missing += ' matplotlib'

    try:
        import scipy
    except ImportError:
        missing += ' scipy'

    if len(missing) > 0:
        if args['install_deps']:
            print("Installing missing dependencies in the user home...")
            os.system("python2 -m pip install --user"+missing)
        else:
            print(
                "Please, install the missing dependencies with `python2 -m pip install [--user]`.")
            print("Missing libraries:", missing)
            sys.exit(2)


def prediction(pr_windows, args, network, func=None):
    if func is None:
        func = network.predict

    prediction = []
    for window in pr_windows:
        # reshaping...
        if args['rnn']:
            r_window = window[np.newaxis, np.newaxis,
                              np.newaxis].astype(settings.floatX)
        else:
            r_window = window[np.newaxis, np.newaxis].astype(settings.floatX)

        p = func(r_window)[0, 0]
        prediction.append(p)

    if args['rnn']:
        prediction = np.array(prediction)
    else:
        prediction = np.array(prediction)[:, np.newaxis, :, :]

    return misc_tools.recreate_pianorolls(prediction, settings.OVERLAP)


def insert_userdir(path):
    """
    this is a utility which returns a new path
    with reference to the environment variable `$USER_PWD`
    as provided by `makeself` command

    If *path* is a list, then the operation is performed over
    all the strings in the list and the list itself will be modified.

    Returns the same object (str or list) but modified.

    If `$USER_PWD` is not set, then the same objects will be returned.

    PARAMETERS :
        path : str or list(str)

    RETURNS :
        str or list(str)
    """
    if 'USER_PWD' in os.environ:
        if path is str:
            path = os.path.join(os.environ['USER_PWD'], path)
        elif path is list:
            for i in range(len(path)):
                path[i] = os.path.join(os.environ['USER_PWD'], path[i])

        else:
            raise Exception("Path must be str or list(str)")

    return path


def prepare_prediction(args, option):
    """
    Prepare stuffs used for prediction.

    PARAMS:
        * `args`: args arrriving from main (command line argument parser)
        * `option`: the name of the option calling this function (`extract`,
        `inspect` or `inspect_masking`)

    RETURNS:
        * `pr_windows`: list of pianoroll windows
        * `network`: the model
        * `notelist`: a list of notes
        * `note_array`: a structured list of notes as in the pickled objects
    """
    # setting default parameters
    model_path = args['model']
    if model_path == DEFAULT_MODEL:
        if args['rnn']:
            model_path = 'rnn_' + DEFAULT_MODEL
        else:
            model_path = 'cnn_' + DEFAULT_MODEL
    else:
        model_path = insert_userdir(model_path)

    insert_userdir(args[option])
    print("Loading file...")
    note_array = parse_data.load_piece(args[option][0], save=False)
    pianoroll, melody, notelist, _notelist_melody = pianoroll_utils.make_pianorolls(
        note_array, output_idxs=True)

    print("Loading the model...")
    network = None
    with open(model_path, 'rb') as f:
        sys.modules['__main__'].cnn = cnn
        network = pickle.load(f)
    if network is None:
        sys.stderr.write("Error, cannot load the neural network model!")
        sys.exit(2)

    WIN_WIDTH = network.win_width

    print("Splitting the input in windows")
    pr_windows = misc_tools.split_windows(
        pianoroll, WIN_WIDTH, settings.OVERLAP)
    mel_windows = misc_tools.split_windows(
        melody, WIN_WIDTH, settings.OVERLAP)

    return (pr_windows, mel_windows), network, notelist, note_array, pianoroll, melody


def extract_solo_part(args):
    windows, network, notelist, note_array, _pianoroll, _melody = prepare_prediction(
        args, 'extract')
    pr_windows = windows[0]

    print("Computing probabilities...")
    out_pianoroll = prediction(pr_windows, args, network)

    _true_labels, predicted_labels = graph_tools.predict_labels(
        out_pianoroll, notelist)

    tracks = (1 - predicted_labels).tolist()
    parse_data.convert_to_midi(
        note_array, tracks=tracks, save=args['extract'][1])


def saliency_masking(inp, network):
    print("Computing original output...")
    inp = inp[np.newaxis, np.newaxis, :, :]
    original_outp = network.predict(inp)

    print("Start masking procedure...")
    mask = np.ones_like(inp)
    output = np.zeros_like(inp)
    correction = np.zeros_like(inp)

    top = 0
    bottom = 128
    ITERATIONS = 30000
    MASK_SIZES = np.array([(2, 4), (4, 8), (8, 16), (16, 32), (16, 8), (32, 8), (4, 2), (8, 4), (32, 64), (64, 32)])
    masked_predictions = []

    for i in range(ITERATIONS):
        if i % 100 == 0:
            print('Iteration ' + str(i))
        N_MASKS = np.random.randint(4)
        mask_idx = np.random.choice(len(MASK_SIZES), size=N_MASKS)

        for MASK_H, MASK_W in MASK_SIZES[mask_idx]:
            i = 0
            if MASK_H < settings.WIN_HEIGHT:
                i = np.random.randint(bottom - MASK_H)
            j = 0
            if MASK_W < settings.WIN_WIDTH:
                j = np.random.randint(settings.WIN_WIDTH - MASK_W)
            mask[:, :, i:i+MASK_H, j:j+MASK_W] = 0
            correction[:, :, i:i+MASK_H, j:j+MASK_W] += 1

        masked_inp = mask * inp
        prediction = network.predict(masked_inp)
        masked_predictions.append(np.stack([prediction[0, 0], mask[0, 0]]))

        output[mask == 0] += np.abs(original_outp - prediction).mean()
        mask[mask == 0] = 1

    output /= np.clip(correction, 1e-15, None, out=correction)
    output /= output.max()
    return output[0, 0], original_outp[0, 0], np.array(masked_predictions)


def inspect_masking(args):
    windows, network, notelist, note_array, in_pianoroll, melody = prepare_prediction(
        args, 'inspect_masking')

    window = windows[0][int(args['inspect_masking'][1])]
    melody = windows[1][int(args['inspect_masking'][1])]

    print("Starting inspection...")
    saliency, out_pianoroll, masked_history = saliency_masking(window, network)

    print("Saving numpy compressed files...")
    np.savez_compressed('inspect.npz',
                    in_pianoroll=window,
                    out_pianoroll=out_pianoroll,
                    melody=melody,
                    saliency=saliency,
                    masked=masked_history
                )

def inspect(args):
    windows, network, notelist, note_array, in_pianoroll, melody = prepare_prediction(
        args, 'inspect')
    pr_windows = windows[0]

    print("Computing probabilities...")
    out_pianoroll = prediction(pr_windows, args, network)

    print("Computing saliency map...")
    saliency = prediction(pr_windows, args, network, func=network.guided_saliency)

    print("Saving numpy compressed files...")
    np.savez_compressed('inspect.npz', in_pianoroll=in_pianoroll,
                        out_pianoroll=out_pianoroll, melody=melody, saliency=saliency)


def train(args):
    from melody_extractor import trainer
    settings.DATA_PATH = insert_userdir(args['train'][0])
    settings.FILE_EXTENSIONS = args['train'][1]
    parameters = json.load(open(insert_userdir(args['train'][2])))

    setup = trainer.setup_train_and_test(parameters)
    _hyperparameters_set, remaining, _groups, X, Y, NN_model, validation, _notelists = setup
    trainer.train(remaining, validation, X, Y, NN_model)
    kernels = NN_model.get_params()
    pickle.dump(kernels, open('nn_kernels_trained.pkl', 'wb'))
    print("Kernels written to file!")


def validate(args):
    from melody_extractor import misc_tools, trainer, graph_tools
    settings.DATA_PATH = insert_userdir(args['validate'][0])
    settings.FILE_EXTENSIONS = args['validate'][1]
    with open(args['validate'][2], 'rb') as f:
        model = pickle.load(f)

    WIN_WIDTH = settings.WIN_WIDTH
    print("Ok, we're ready to load files, let's start!")
    X, Y, map_sw, notelists = misc_tools.load_files(
        settings.DATA_PATH, WIN_WIDTH, return_notelists=True, overlap=True)
    data, groups = misc_tools.build_groups(map_sw)
    pieces_indices = np.unique(groups)

    prediction_list = trainer.predict(data, groups, X, model)

    graph_tools.test_shortest_path(notelists,
                                   prediction_list,
                                   pieces_indices,
                                   open("validation.txt", "w"))

    print("Starting predictions...")


def crossvalidate(args):
    import melody_extractor.crossvalidation as cv
    settings.DATA_PATH = insert_userdir(args['crossvalidation'][0])
    settings.FILE_EXTENSIONS = args['crossvalidation'][1]
    parameters = json.load(open(insert_userdir(args['crossvalidation'][2])))

    cv.crossvalidation(parameters)


def rebuild(args):
    from melody_extractor import trainer
    insert_userdir(args['rebuild'])
    with open(args['rebuild'][0], 'rb') as f:
        kernels = pickle.load(f)
    parameters = json.load(open(args['rebuild'][1]))

    if settings.MODEL_TYPE == 'cnn':
        NN_model = trainer.build_CNN_model(parameters)
    else:
        args['gradient_steps'] = int(min([len(i) for i in map_sw]))
        print("Gradient steps: " + str(args['gradient_steps']))
        NN_model = trainer.build_RNN_model(parameters)

    NN_model.set_params(kernels)

    with open(args['rebuild'][2], 'wb') as f:
        pickle.dump(NN_model, f)


def hyperopt(args):
    from melody_extractor import trainer
    settings.DATA_PATH = insert_userdir(args['hyper_opt'][0])
    settings.FILE_EXTENSIONS = args['hyper_opt'][1]
    trainer.hyperopt(parameters_path=insert_userdir(args['hyper_opt'][2]))


def main():
    args = check_args()

    # checking libs
    if args['install_deps'] or args['check_deps']:
        check_libraries(args)
        return

    # setting global variables
    if args['mono']:
        settings.MONOPHONIC = True

    if args['rnn']:
        settings.MODEL_TYPE = 'rnn'
        settings.OVERLAP = False
    else:
        settings.MODEL_TYPE = 'cnn'
        settings.OVERLAP = True

    settings.NUM_EPOCHS = args['epochs']

    settings.MAX_TIME = args['time_limit']

    if len(args['extract']) == 2:
        extract_solo_part(args)
        return

    if len(args['inspect']) == 1:
        inspect(args)
        return

    if len(args['train']) == 3:
        train(args)
        return

    if len(args['crossvalidation']) == 3:
        crossvalidate(args)
        return

    if len(args['hyper_opt']) == 3:
        hyperopt(args)
        return

    if len(args['rebuild']) == 3:
        rebuild(args)
        return

    if len(args['validate']) == 3:
        validate(args)
        return

    if len(args['inspect_masking']) == 2:
        inspect_masking(args)
        return

    print("specify at least one action and all arguments needed.")
    print("use -h or no option for help!")


main()
