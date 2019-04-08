"""
simply store settings variables
"""
import numpy as np
import lasagne.updates
import lasagne.nonlinearities
import lasagne.objectives
import lasagne.init
from hyperopt import hp
import hyperopt
floatX = np.float32

# model type used for the network: 'cnn' or 'rnn'
MODEL_TYPE = 'cnn'

# a tuple containing the extensions of used files
FILE_EXTENSIONS = ('.pyc.bz')

THRESHOLD = 0.5  # threshold used in computing the graph matrix
# with *THRESHOLD* <= 0.0 it's like removing it

# if `CLUSTERING` is `None`, THRESHOLD will be used, otherwise it will be setted
# through clustering; in that case, 'kmeans' causes the algorithm to use kmeans,
# 'single' single-linkage, 'average' average-linkage, 'centroid' centroid-linkage
CLUSTERING = 'centroid'

# these are used to filter out samples before of clustering so that it's faster
# shouldn't be touched until great changes are apported or repertory is changed
MIN_TH = 1e-15

# if true, when computing the probability of a note, we try to remove the pixels
# that are outliers
OUTLIERS_ON_PROB = False

# If True, the probability of a note is computed as the average probability of
# its pixels, otherwise as the median
AVERAGE = False

# algorithm for shortest path, according ot
# scipy.sparse.csgraph.shortest_path([...])
PATH_METHOD = 'BF'

# If the following is True, then just one note at a time is considered as melody
MONOPHONIC = False

# This is the maximum loss accepted: if the validation loss is bigger than this,
# than nn_models.cnn.fit(...) launches a runtime error and the training stops. It is
# particularly useful during hyperoptimization, to discard probably bad
# combinations of parameters

# As of now, it the loss function is bigger than this after the first epoch,
# targets will be switched
MAX_LOSS_ACCEPTED = 0.5

# this is the number of evaluations performed by the hyper-optimization
# procedure
EVALS = 300000
# the algorithm used to optimize: `hyperopt.tpe.suggest` or `hyperopt.rand.suggest`
SUGGEST = hyperopt.rand.suggest
DATA_PATH = "./data/mcleod_comparison/"
HYPEROPT_PATH = "../data/hyper-opt"
WIN_HEIGHT = 128
WIN_WIDTH = 64
# the maximum time allowed for an epoch in seconds. if time needed for an epoch
# is bigger than this, nn_models.cnn.fit(...) launches a RuntimeError, the
# training is interrupted and the parameters with the best loss are used,
# eventually the initial ones
MAX_TIME = 120

# this is the maximum number of epochs, but training uses early-stopping
NUM_EPOCHS = 5000
NONLINEARITY = lasagne.nonlinearities.sigmoid

# the loss function used to train and validate
LOSS = lasagne.objectives.squared_error

# the function for the updates
UPDATE = lasagne.updates.adadelta

# the batch size
BATCH_SIZE = 100 # (100, 0.05)

# use this for debugging purposes: load just this percentage of the dataset
DATASET_PERC = 1.0

# set to False to skip data augmentation
DATA_AUGMENTATION = True

# set to false to skip the masking with the input
MASKED = True

# set to false to skip the initialization through autoencoders
AUTOENCODERS = False

# if the following is True, then during hyperparameter optimization will be
# used 10-fold cross validation
HYPERPARAMS_CROSS_VALIDATION = False # True

# the number of preliminary networks before of the output one (only CNN)
PRELIMINARY_CNN = 2

# CNN_DEFAULT = {
#     'maxpool_h': 6.0,
#     'kernel_w': 63.0,
#     'dropout_p': 0.2948569243811522,
#     'win_w': 512,
#     'learning_rate': 0.2773263017254355,
#     'num_kernel': 28.0,
#     'maxpool_w': 15.0
# }

# This is the space of variables for hyperopt
RNN_SPACE = hp.choice('x',
                      [{
                          'win_w': hp.choice('win_w', [32, 64, 128, 256]),
                          'num_kernel': hp.quniform('num_kernel', 1, 30, 6),
                          'kernel_w': hp.quniform('kernel_w', 4, 128, 8),
                          'dropout_p': hp.choice('dropout_p', [0.2, 0.4, 0.3, 0.5]),
                          'maxpool_h': hp.quniform('maxpool_h', 1, 128, 8),
                          'maxpool_w': hp.quniform('maxpool_w', 1, 128, 8),
                          'learning_rate': hp.choice('learning_rate', [0.001, 0.01, 0.1]),
                          'lstm_units': hp.quniform('lstm_units', 50, 250, 25)
                      }]
                      )

CNN_SPACE = hp.choice('x',
                      [{
                          #
                          # 'win_w': hp.choice('win_w', [32, 64, 96, 128]),
                          'kernel_h0': hp.choice('kernel_h0', [128, 64, 32]),
                          'kernel_w0': hp.choice('kernel_w0', [WIN_WIDTH, WIN_WIDTH / 2, WIN_WIDTH / 4]),
                          'num_kernel0': hp.quniform('num_kernel0', 3, 60, 3),
                          # 'maxpool_h0': hp.quniform('maxpool_h0', 1, 30, 3),
                          # 'maxpool_w0': hp.quniform('maxpool_w0', 1, 30, 3),
                          # 'learning_rate': hp.uniform('learning_rate', 0.00001, 0.0001),
                          # 'dropout_p': hp.uniform('dropout_p', 0.1, 0.5),

                          # 'kernel_w1': hp.quniform('kernel_w1', 31, 128, 4),
                          # 'maxpool_h1': hp.quniform('maxpool_h1', 1, 30, 3),
                          # 'maxpool_w1': hp.quniform('maxpool_w1', 1, 30, 3),
                          # 'num_kernel1': hp.quniform('num_kernel1', 20, 60, 4),

                          # 'kernel_w2': hp.quniform('kernel_w2', 31, 128, 4),
                          # 'maxpool_h2': hp.quniform('maxpool_h2', 1, 30, 3),
                          # 'maxpool_w2': hp.quniform('maxpool_w2', 1, 30, 3),
                          # 'num_kernel2': hp.quniform('num_kernel2', 20, 60, 4),

                          # 'preliminary_cnns': hp.choice('preliminary_cnns', [0]),
                          # 'initializer': hp.choice('initializer', ['normal', 'uniform', 'glorotnormal', 'glorotuniform', 'henormal', 'heuniform']),
                          # 'num_layers': hp.choice('num_layers', [1]) #, 2, 3]),
                      }]
                      )
