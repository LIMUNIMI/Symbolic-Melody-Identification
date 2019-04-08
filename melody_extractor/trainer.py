import argparse
import os
import json

import lasagne
import numpy as np
import sklearn.metrics
import theano.tensor as T
from hyperopt import STATUS_OK, STATUS_FAIL, fmin, Trials
from sklearn.model_selection import train_test_split, GroupShuffleSplit

import misc_tools
import settings
import graph_tools
from nn_models import helper
from nn_models.rnn import RNN
from nn_models.cnn import CNN
import crossvalidation as cv


try:
    import cPickle as pickle
except Exception:
    import pickle

OUT_FILE = "global variable to contain output path of intermediate results"


def run_trials(objective, trials):

    if settings.MODEL_TYPE == 'cnn':
        space = settings.CNN_SPACE
    else:
        space = settings.RNN_SPACE

    best = 1
    max_trials = len(trials.trials) + 1
    try:
        best = fmin(fn=objective, space=space, algo=settings.SUGGEST,
                    max_evals=max_trials, trials=trials)
    except Exception:
        import traceback
        traceback.print_exc()

    # save the trials object
    with open("trials.hyperopt", "wb") as f:
        pickle.dump(trials, f)
    print("saved trials object")

    return best


def hyperopt(parameters_path='parameters.json'):
    """
    Perform the hyper-parameter optimization in files in
    `settings.HYPEROPT_PATH` directory (and subdir too). Only files compliant to
    `settings.FILE_EXTENSIONS` are considered. Save best parameters on each evaluation
    in 'parameters.json' by default, otherwise to the path specified as argument.

    Moreover, saves 'trials.hyperopt' object containing the evaluation records to be re-used
    if the hyper-optimization stops for any cause.
    """

    global OUT_FILE
    OUT_FILE = open("hyperoptimization.txt", "w")

    # settings.DATASET_PERC = 0.15
    print("HYPER-OPTIMIZATION")
    try:  # try to load an already saved trials object, and increase the max
        trials = pickle.load(open("trials.hyperopt", "rb"))
        print("Found saved Trials! Loading...")
    except Exception:  # create a new trials object and start searching
        trials = Trials()

    for i in range(settings.EVALS - len(trials.trials)):
        print("Evaluation number " + str(i + 1))
        best = run_trials(objective, trials)
        # saving the best parameters
        if best != 1:
            print("BEST PARAMETERS FOOUND: " + str(best))
            json.dump(best, open(parameters_path, 'w'))

    print("________________")
    print("BEST  PARAMETERS")
    print(best)


def data_augmentation(indices, X, Y):
    """ Perform data augmentation by lowering down the melody in the 50% of the windows
    listed in *indices*.

    PARAMETERS :
        X : numpy.ndarray with shape (num_of_windows, 1, window_height, window_length)
            this is the array containing the input windows (melodies + accompaniment)
        Y : numpy.ndarray with shape (num_of_windows, 1, window_height, window_length)
            this is the array containing the output windows (melodies, ground truth)
        indices : an array-like containing the indices from which the 50% will be
            randomly extracted and used for data augmentation

    RETURNS :
        numpy.ndarray with shape (int(len(indices) / 2), 1, window_height, window_length)
            a numpy array containing the generated windows
    """

    # taking the 50% of windows
    np.random.seed(1992)
    # extracted_indices = np.random.randint(
    #     len(indices), size=int(len(indices)))
    extracted_indices = np.random.randint(
        len(indices), size=int(len(indices) / 2))

    Xreturned = []
    Yreturned = []

    for i in extracted_indices:
        # for each extracted window
        pianoroll = X[i].copy()
        melody = Y[i].copy()

        # take melody_indices
        idx = np.argwhere(melody > 0.5)

        # for each melody index
        for j in idx:
            # remove the melody
            melody[j[0], j[1], j[2]] = 0
            pianoroll[j[0], j[1], j[2]] = 0

            # re-add the melody 1 or 2 octaves lower or 1 octave higher
            k = np.random.randint(0, 3)
            # if k == 0:
            #     new_pitch = min(j[2] + 12, 255)
            # else:
            new_pitch = max(j[2] - k * 12, 0)
            melody[j[0], j[1], new_pitch] = 1
            pianoroll[j[0], j[1], new_pitch] = 1

        Xreturned.append(pianoroll)
        Yreturned.append(pianoroll)

    return np.array(Xreturned), np.array(Yreturned)


def train(training, validation, X, Y, NN_model, nan_exception=False):
    """ trains a NN_model
    If `settings.DATA_AUGMENTATION` is *True*, then a data augmentation is performed
    by transposing down the melody in the 50% of the windows.

    PARAMETERS :
    ------------
        *training * is an array-like of indices in X and Y
        *validation * is another array-like of indices in X and Y
        *X * is a 4D array containing input windows
        *Y * is a 4D array containing output windows(ground truth)
        *NN_model * the model that should be trained, with `fit` method
            compliant to the one of nn_models.cnn.CNN and nn_models.rnn.RNN
    """

    if len(training) <= len(validation):
        print("Validation set bigger than training set, skipping it!")
        return 'error'

    if settings.DATA_AUGMENTATION:
        addedX, addedY = data_augmentation(training, X, Y)
        added_indices = range(X.shape[0], addedX.shape[0] + X.shape[0])

        # adding the augmented windows
        training = np.concatenate((training, added_indices))
        X = np.concatenate((X, addedX))
        Y = np.concatenate((Y, addedY))

    # if settings.MODEL_TYPE == 'cnn':
    #     BATCH_PERC = 0.05
    # else:
    #     BATCH_PERC = 0.1
    BATCHSIZE =  settings.BATCH_SIZE # max(settings.BATCH_SIZE[0], int(len(training) * settings.BATCH_SIZE[1]))
    print("BATCH SIZE: " + str(BATCHSIZE))

    # shuffle them
    np.random.seed(34)
    np.random.shuffle(training)
    np.random.seed(78)
    np.random.shuffle(validation)

    # only CNN for now...
    if settings.AUTOENCODERS:
        print("Looking for initial parameters through autoencoders")
        print("This could take a while... using 200 epochs")
        NN_model.fit(X, X, training, val_map=validation,
                     NUM_EPOCHS=200, BATCHSIZE=BATCHSIZE,
                     nan_exception=nan_exception, masked=False)

    print("---------------------------------")
    print("And now train the network...")
    print("...sorry, this could take a while...")
    # train(training, validation, X, Y, val_fn, train_fn)
    NN_model.fit(
        X=X,
        Y=Y,
        tr_map=training,
        val_map=validation,
        NUM_EPOCHS=settings.NUM_EPOCHS,
        BATCHSIZE=BATCHSIZE,
        nan_exception=nan_exception,
        masked=settings.MASKED
    )


def chose_initializer(label):
    """
    return a Python binding to a function in lasagne.init according to the label passed:
        `normal` -> lasagne.init.Normal
        `uniform` -> lasagne.init.Uniform
        `glorotnormal` -> lasagne.init.GlorotNormal
        `glorotuniform` -> lasagne.init.GlorotUniform
        `heuniform` -> lasagne.init.HeUniform
        `henormal` -> lasagne.init.HeNormal
    """
    if label == 'normal':
        return lasagne.init.Normal
    elif label == 'uniform':
        return lasagne.init.Uniform
    elif label == 'glorotnormal':
        return lasagne.init.GlorotNormal
    elif label == 'glorotuniform':
        return lasagne.init.GlorotUniform
    elif label == 'heuniform':
        return lasagne.init.HeUniform
    elif label == 'henormal':
        return lasagne.init.HeNormal


def build_CNN_model(args, WIN_HEIGHT=settings.WIN_HEIGHT, NONLINEARITY=settings.NONLINEARITY, UPDATE=settings.UPDATE):
    """ returns a cnn.ConvolutionalNeuralNetwork object built with
    hyper-parameters contained in *args*
    This also stores the WIN_WIDTH in the field `win_width` of the model"""

    WIN_WIDTH = settings.WIN_WIDTH
    NUM_KERNEL = [int(args['num_kernel0']), ] # , int(args['num_kernel1']), int(args['num_kernel2'])]
    KERNEL_H = [int(args['kernel_h0'])]
    KERNEL_W = [int(args['kernel_w0'])] # , int(args['kernel_w1']) - 1, int(args['kernel_w2']) - 1]
    # KERNEL_H = KERNEL_W
    # MAXPOOL_H = [int(args['maxpool_h0'])] # , int(args['maxpool_h1']), int(args['maxpool_h2'])]
    # MAXPOOL_W = [int(args['maxpool_w0'])] # , int(args['maxpool_w1']), int(args['maxpool_w2'])]
    DROPOUT_P = 0.3  # args['dropout_p']
    LEARNING_RATE = 1 # 1e-2  # args['learning_rate']
    NUM_LAYERS = 1 # int(args['num_layers'])
    PRELIMINARY_CNN = 0 # int(args['preliminary_cnns'])
    INITIALIZER = lasagne.init.GlorotUniform # chose_initializer(args['initializer'])

    # NUM_LAYERS = 1

    print("Hi all! I'm gonna start to build the network...")
    l_in = lasagne.layers.InputLayer((None, 1, WIN_HEIGHT, WIN_WIDTH))

    loss = settings.LOSS

    target = T.tensor4()

    input_l = l_in
    for i in range(PRELIMINARY_CNN):
        # the preliminary CNNs without masking
        cnn_i = CNN(input_l, l_in, loss, loss, settings.UPDATE, target,
                    NUM_KERNEL[i], KERNEL_H[i], KERNEL_W[i],
                    NUM_LAYERS, NONLINEARITY,
                    DROPOUT_P, LEARNING_RATE, INITIALIZER, MASKING=False, TYPE=None)
        input_l = cnn_i.l_out

    i = PRELIMINARY_CNN
    # the output CNN with masking
    cnn = CNN(input_l, l_in, loss, loss, settings.UPDATE, target,
              NUM_KERNEL[i], KERNEL_H[i], KERNEL_W[i],
              NUM_LAYERS, NONLINEARITY,
              DROPOUT_P, LEARNING_RATE, INITIALIZER, MASKING=True, TYPE='output')

    cnn.win_width = WIN_WIDTH
    return cnn


def build_RNN_model(args, WIN_HEIGHT=settings.WIN_HEIGHT, NONLINEARITY=settings.NONLINEARITY):
    """
    returns a nn_models.models.RNN object built with
    hyper-parameters contained in *args*
    This also stores the WIN_WIDTH in the field `win_width` of the model
    """

    WIN_WIDTH = int(args['win_w'])
    NUM_KERNEL = int(args['num_kernel'])
    KERNEL_W = int(args['kernel_w'])
    KERNEL_H = KERNEL_W
    MAXPOOL_H = int(args['maxpool_h'])
    MAXPOOL_W = int(args['maxpool_w'])
    DROPOUT_P = args['dropout_p']
    LEARNING_RATE = args['learning_rate']
    NUM_UNITS = int(args['lstm_units'])
    NUM_LAYERS = 1
    NUM_CHANNELS = 1
    GRADIENT_STEPS = int(args['gradient_steps'])

    print("Hi all! I'm gonna start to build the network...")
    rcnn_kwargs = helper.get_rcnn_kwargs(WIN_WIDTH,
                                         WIN_HEIGHT,
                                         NUM_CHANNELS,
                                         NUM_KERNEL,
                                         (KERNEL_H, KERNEL_W),
                                         (MAXPOOL_H, MAXPOOL_W),
                                         NUM_UNITS,
                                         GRADIENT_STEPS)

    rcnn_kwargs.pop('model_type')
    model = RNN(**rcnn_kwargs)
    model.win_width = WIN_WIDTH
    return model


def predict(testing, groups, X, NN_model):
    # Reordering testing and groups according to groups
    f = np.rec.fromarrays([groups, testing])
    f.sort()
    groups = f.f0
    testing = f.f1

    predictions = []
    prediction = []
    for i, w_index in enumerate(testing):
        x = X[w_index]
        if settings.MODEL_TYPE == 'cnn':
            p = NN_model.predict(x[np.newaxis])[0, 0]
        else:
            p = NN_model.predict(x[np.newaxis, np.newaxis])[0, 0]
        prediction.append(p * x[0])

        if i + 1 == len(testing) or groups[i] != groups[i + 1]:
            if settings.MODEL_TYPE == 'cnn':
                prediction = np.array(prediction)[:, np.newaxis, :, :]
                t = misc_tools.recreate_pianorolls(
                    prediction, overlap=True)
            else:
                prediction = np.array(prediction)
                t = misc_tools.recreate_pianorolls(
                    prediction, overlap=False)
            predictions.append(t)
            prediction = []
    return predictions


def simple_validation(args):
    """
    This performs a training and testing over the * perc * of the whole
    dataset. It saves intermediate results in the global `OUT_FILE` file object.
    It uses a fixed random seed, so that successive calls will produce the same
    results. Returns the average fmeasure.

    """

    nan_exception = 'nan_exception' in args

    hyperparameters_set, remaining, groups, X, Y, NN_model, testing, notelists = setup_train_and_test(
        args, random_state=1992)

    fmeasures, precisions, recalls = train_and_test(
        hyperparameters_set, remaining, groups, X, Y, NN_model, testing, notelists, OUT_FILE=OUT_FILE, nan_exception=nan_exception)
    print("F-measures: " + str(fmeasures))
    print("Precisions: " + str(precisions))
    print("Recalls: " + str(recalls))
    average_fmeasure = np.mean(fmeasures)
    OUT_FILE.write(
        "\nAvarage F1-measure: " + str(average_fmeasure) + "\n")
    OUT_FILE.flush()
    if np.isnan(average_fmeasure):
        raise Exception(
            "NaN in fmeasure - probably some no predictions were made")
    return average_fmeasure


def setup_train_and_test(args, random_state=1992):
    """
    Returns variables to be used in train_and_test function
    """
    if settings.MODEL_TYPE == 'cnn':
        OVERLAP = True
    else:
        OVERLAP = False

    WIN_WIDTH = settings.WIN_WIDTH
    print("Ok, we're ready to load files, let's start!")
    X, Y, map_sw, notelists = misc_tools.load_files(
        settings.DATA_PATH, WIN_WIDTH, return_notelists=True, overlap=OVERLAP)

    print("Separating training, validation and test set...")

    if settings.MODEL_TYPE == 'cnn':
        NN_model = build_CNN_model(args)
    else:
        args['gradient_steps'] = int(min([len(i) for i in map_sw])) / 2
        NN_model = build_RNN_model(args)

    # building groups
    hyperparameters_set, groups = misc_tools.build_groups(map_sw)
    test_size = max(0.2, 30.0 / len(np.unique(groups)))
    # if data are very very little, use 0.2
    if test_size >= 0.5:
        test_size = 0.2
    print("using test_size = " + str(test_size))
    remaining, testing = GroupShuffleSplit(test_size=test_size, random_state=random_state).split(
        hyperparameters_set, groups=groups).next()
    return hyperparameters_set, remaining, groups, X, Y, NN_model, testing, notelists


def train_and_test(dataset, remaining, groups, X, Y, NN_model, testing, notelists, OUT_FILE=None, nan_exception=False):
    """
    Perform a training and a test. Returns fmeasures precisions and recalls on each group.
    """
    val_size = max(0.2, 30.0 / len(np.unique(groups[remaining])))
    # if data are very very little, use 0.2
    if val_size >= 0.5:
        val_size = 0.2
    print("using val_size = " + str(val_size))
    training, validation = GroupShuffleSplit(test_size=val_size, random_state=34).split(
        dataset[remaining], groups=groups[remaining]).next()

    train(dataset[training],
          dataset[validation], X, Y, NN_model, nan_exception)
    print("And test!")

    groups_testing = groups[testing]
    test_notelists = notelists[np.unique(groups_testing)]
    if OUT_FILE:
        pieces_indices = np.unique(groups_testing)
    else:
        pieces_indices = None

    prediction_list = predict(testing, groups_testing, X, NN_model)
    fmeasures, precisions, recalls = graph_tools.test_shortest_path(test_notelists,
                                                                    prediction_list,
                                                                    pieces_indices,
                                                                    OUT_FILE)
    return fmeasures, precisions, recalls


def objective(args):
    """ This function can be used with hyperopt module to optimize
    the hyperparameters of the model

    """
    outstr = "\nTESTING THE FOLLOWING PARAMETERS: " + str(args)
    OUT_FILE.write(outstr)
    OUT_FILE.flush()
    print(outstr)

    args['nan_exception'] = True

    try:
        if settings.HYPERPARAMS_CROSS_VALIDATION:
            loss = 1 - cv.crossvalidation(args, OUT_FILE=OUT_FILE)
        else:
            loss = 1 - simple_validation(args)
        return {'loss': loss, 'status': STATUS_OK}

    except Exception:
        print("Error occured")
        import traceback
        traceback.print_exc()
        return {'loss': 2, 'status': STATUS_FAIL}


if __name__ == '__main__':
    import theano.gpuarray
    theano.gpuarray.use('cuda1')

    hyperopt()
