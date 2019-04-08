"""
This uses `trainer` and `melody_identification` to perform a whole
10-fold crossvalidation, including NN_model training, validation and testing through
Djikstra algorithm for DAG graphs with non-negative weigths
"""
from copy import deepcopy

from sklearn.model_selection import GroupShuffleSplit
import numpy as np

import graph_tools
import trainer
import settings
import misc_tools
import math

try:
    import cPickle as pickle
except ImportError:
    import pickle


def crossvalidation(args, OUT_FILE=open("crossvalidation.txt", "w")):
    """
    This performs a 10-fold cross-validation, included graph test and saves
    results in the global `OUT_FILE` file object.
    """
    if settings.MODEL_TYPE == 'cnn':
        OVERLAP = True
    else:
        OVERLAP = False

    print("Ok, we're ready to load files, let's start!")
    X, Y, map_sw, notelists = misc_tools.load_files(
        settings.DATA_PATH, return_notelists=True, overlap=OVERLAP)

    if settings.MODEL_TYPE == 'cnn':
        NN_model = trainer.build_CNN_model(args)
    else:
        args['gradient_steps'] = int(min([len(i) for i in map_sw]))
        print("Gradient steps: " + str(args['gradient_steps']))
        NN_model = trainer.build_RNN_model(args)
    initial_params = NN_model.get_params()

    print("10 CROSS-VALIDATION")
    print("Separating training, validation and test set...")

    from sklearn.model_selection import GroupKFold
    kfold = GroupKFold(n_splits=10)
    k = 1

    dataset, groups = misc_tools.build_groups(map_sw)

    fmeasures = []
    recalls = []
    precisions = []
    for remaining, testing in kfold.split(dataset, groups=groups):
        print("")
        print('Training on fold number ' + str(k))
        NN_model.set_params(initial_params)
        fmeasure_chunk, precision_chunk, recall_chunk = trainer.train_and_test(
            dataset, remaining, groups, X, Y, NN_model, testing, notelists, OUT_FILE)

        parameters = NN_model.get_params()
        with open('nn_kernels_' + str(k) + '.pkl', 'wb') as f:
            pickle.dump(parameters, f)
        print("Parameters written to file!")

        fmeasures += fmeasure_chunk
        recalls += recall_chunk
        precisions += precision_chunk
        k += 1

        # if results contain a nan throw an exception
        if any([math.isnan(i) for i in fmeasure_chunk]):
            raise Exception("nan in crossvalidation step" + str(k))

    OUT_FILE.write("\nAverage precision: " + str(np.mean(precisions)))
    OUT_FILE.write("\nAverage recall: " + str(np.mean(recalls)))
    OUT_FILE.write("\nAverage fmeasure: " + str(np.mean(fmeasures)) + "\n")
    OUT_FILE.flush()

    return np.mean(fmeasures)


if __name__ == '__main__':
    crossvalidation(settings.ARGS_DEFAULT)
