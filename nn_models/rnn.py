import numpy as np
import theano
import lasagne
import logging
from itertools import cycle
import os
import cPickle
import bz2

from batch_provider import RecurrentBatchProvider
from melody_extractor import settings


logging.basicConfig()
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

# Set random state for all random initializations in Lasagne
lasagne.random.set_rng(np.random.RandomState(1984))


def ensure_list(input):
    if not isinstance(input, (list, tuple)):
        return [input]
    else:
        return input


def save_pyc_bz(d, fn):
    # Copied from extra.utils.os_utils
    cPickle.dump(d, bz2.BZ2File(fn, 'w'), cPickle.HIGHEST_PROTOCOL)


def delete_if_exists(fn):
    """Delete file if exists
    """
    try:
        os.unlink(fn)
    except OSError:
        pass


def write_append(fn, epoch, value, fmt='{0} {1:.8f}\n'):
    """Append value to a file
    """
    with open(fn, 'a') as f:
        f.write(fmt.format(epoch, value))


class NeuralNetwork(object):

    """
    Base class for training neural networks
    """

    def __init__(self, l_in, l_out,
                 train_loss,
                 valid_loss,
                 target,
                 input_vars=None,
                 updates=None,
                 predictions=None,
                 input_names=None,
                 output_names=None,
                 output_dim=None,
                 n_components=None,
                 input_type=None,
                 out_dir='/tmp',
                 visualize=False,
                 random_state=np.random.RandomState(1984)):

        # Input layer
        self.l_in = l_in
        # Output layer
        self.l_out = l_out
        # Target
        self.target = target
        # Input variables
        self.input_vars = input_vars

        if predictions is None:
            predictions = lasagne.layers.get_output(
                self.l_out, determistic=True)

        # Set name of the input features
        self.input_names = input_names
        # Set name of the output features
        self.output_names = output_names
        # Set description of the input
        self.input_type = input_type
        # Set path to save output files
        self.out_dir = out_dir

        # Random number generator
        self.prng = random_state

        # Initialize as model as non-recurrent
        self.is_rnn = False

        self._setup_graphs(train_loss, valid_loss,
                           predictions, target, updates)

    def _setup_graphs(self, stochastic_loss, deterministic_loss,
                      predictions, target, updates):

        if self.input_vars is None:
            input_vars_loss = [self.l_in.input_var, target]
            input_vars_predict = [self.l_in.input_var]
        else:
            input_vars_loss = ensure_list(self.input_vars) + [target]
            input_vars_predict = ensure_list(self.input_vars)

        if isinstance(updates, (list, tuple)):
            # train_fun is a list for all updates
            self.train_fun = [theano.function(
                inputs=input_vars_loss,
                outputs=ensure_list(stochastic_loss),
                updates=up) for up in updates]
        else:
            self.train_fun = theano.function(
                inputs=input_vars_loss,
                outputs=ensure_list(stochastic_loss),
                updates=updates)

        self.valid_fun = theano.function(
            inputs=input_vars_loss,
            outputs=ensure_list(deterministic_loss))

        if any(p is None for p in ensure_list(predictions)):
            preds = []

            for fun in ensure_list(predictions):
                if fun is None:
                    preds.append(lambda x: None)
                else:
                    preds.append(theano.function(
                        inputs=input_vars_predict,
                        outputs=fun))

            self.predict_fun = lambda x: [pr(x) for pr in preds]
        else:
            self.predict_fun = theano.function(
                inputs=input_vars_predict,
                outputs=ensure_list(predictions))

    def fit(self, *args, **kwargs):
        raise NotImplementedError("NeuralNetwork does not "
                                  "implement the fit method")

    def get_params(self):
        """
        Get the parameters of the network.

        Returns
        -------
        iterable of arrays
            A list with the values of all parameters that specify the
            neural network.
        """
        return lasagne.layers.get_all_param_values(self.l_out)

    def set_params(self, params):
        """
        Set the parameters of the neural network

        Parameters
        ----------
        params : iterable of arrays
            An iterable that contains the values of all parameters that
            specify the neural network. Each parameters must be of the
            same size and dtype as the original parameters.
        """
        return lasagne.layers.set_all_param_values(self.l_out, params)

    def predict(self, X):
        """
        Compute predictions of the neural network for the given input.

        Parameters
        ----------
        X : array
            Input array for the neural network.

        Returns
        -------
        array
            Predictions of the neural network.
        """

        return self.predict_fun(X)[0]

    def get_param_dict(self):
        params = lasagne.layers.get_all_params(self.l_out)
        self.param_dict = dict([(p.name, p) for p in params])

    def set_param_dict(self, param_dict):
        if not hasattr(self, 'param_dict'):
            self.get_param_dict()

        for p in param_dict:
            if p in self.param_dict:
                print 'Using {0}'.format(p)
                self.param_dict[p].set_value(param_dict[p].get_value())
            else:
                print '{0} not in model'.format(p)


class RNN(NeuralNetwork):
    """
    Class for training Recurrent Neural Networks
    """

    def __init__(self, *args, **kwargs):

        self.args = args

        if 'gradient_steps' in kwargs:
            self.gradient_steps = kwargs.pop('gradient_steps')

        self.kwargs = kwargs

        self._build_model()

    def _build_model(self):
        super(RNN, self).__init__(
            *self.args, **self.kwargs)
        self.is_rnn = True

    def __getstate__(self):
        """
        Get current state of the object for pickling.

        Returns
        -------
        dict
            A dictionary containing all parameters to fully reconstruct
            an instance of the class.
        """
        state = dict(
            args=self.args,
            kwargs=self.kwargs,
            params=self.get_params(),
            gradient_steps=self.gradient_steps
        )
        return state

    def __setstate__(self, state):
        """
        Set a state for unpickling the object.

        Parameters
        ----------
        state : dict
            A pickled dictionary containing the state to reconstruct the
            :class:`NeuralNetwork` instance. Valid states are
            generated with `__getstate__`.
        """
        self.gradient_steps = state['gradient_steps']
        self.args = state['args']
        self.kwargs = state['kwargs']
        self._build_model()
        self.set_params(state['params'])

    def fit(self, X, Y,
            tr_map, val_map=None,
            n_epochs=100, batch_size=10,
            max_epochs_from_best=20,
            keep_training=False,
            nan_exception=False):

        if self.gradient_steps < 1:
            seq_length = 1
        else:
            seq_length = 2 * self.gradient_steps

        start_epoch = 0
        best_epoch = 0
        best_loss = np.inf
        best_params = self.get_params()
        validate = val_map is not None

        train_loss_fn = os.path.join(self.out_dir, 'train_loss.txt')

        if not keep_training:
            delete_if_exists(train_loss_fn)

        if validate:
            valid_loss_fn = os.path.join(self.out_dir, 'valid_loss_reg.txt')
            valid_loss_2_fn = os.path.join(self.out_dir, 'valid_loss.txt')
            if not keep_training:
                delete_if_exists(valid_loss_fn)
                delete_if_exists(valid_loss_2_fn)

            else:
                if os.path.exists(valid_loss_fn):
                    valid_loss_old = np.loadtxt(valid_loss_fn)
                    best_loss_idx = np.argmin(valid_loss_old[:, 1])
                    best_loss = valid_loss_old[best_loss_idx, 1]
                    best_epoch = valid_loss_old[best_loss_idx, 0]
                    start_epoch = int(best_epoch) + 1

                elif os.path.exists(train_loss_fn):
                    train_loss_old = np.loadtxt(train_loss_fn)
                    best_loss_idx = np.argmin(train_loss_old[:, 1])
                    best_loss = train_loss_old[best_loss_idx, 1]
                    best_epoch = train_loss_old[best_loss_idx, 0]
                    start_epoch = int(best_epoch) + 1

        else:
            named_valid_results = ()

        train_batch_provider = RecurrentBatchProvider(settings.floatX)
        train_batch_provider.store_data(
            X[tr_map, :, np.newaxis, :, :], Y[tr_map, :, np.newaxis, :, :])

        valid_batch_provider = RecurrentBatchProvider(settings.floatX)
        valid_batch_provider.store_data(
            X[val_map, :, np.newaxis, :, :], Y[val_map, :, np.newaxis, :, :])

        total_train_instances = len(tr_map)
        n_train_batches_per_epoch = max(
            1, 2 * total_train_instances / (batch_size * seq_length))

        LOGGER.info('Batch size: {}; Batches per epoch: {}'
                    .format(batch_size, n_train_batches_per_epoch))

        # variables to hold data batches; reusing rather than recreating the
        # arrays saves (a little bit of) time
        batch_arrays = train_batch_provider.make_batch_arrays(
            batch_size, seq_length)
        # y_t = train_batch_provider.make_Y_batch_array(batch_size, seq_length)

        if isinstance(self.train_fun, (tuple, list)):
            train_mode_selector = ParameterUpdate(start_epoch, n_epochs)
        else:
            train_mode_selector = SimpleParameterUpdate()

        # Initialize valid loss (in case there is no validation set)
        valid_loss = np.array([0 for o in self.valid_fun.outputs])

        try:
            for epoch in xrange(start_epoch,
                                n_epochs):

                # train_results = []
                mode = train_mode_selector.select_mode(epoch)
                LOGGER.info('Training {0} params'.format(mode))
                train_loss, named_train_results = _train_loss(
                    self,
                    batch_provider=train_batch_provider,
                    batch_size=batch_size,
                    seq_length=seq_length,
                    X_t=batch_arrays,
                    y_t=None,
                    train_loss_fn=train_loss_fn,
                    epoch=epoch,
                    batches_per_epoch=n_train_batches_per_epoch,
                    mode=mode,
                    nan_exception=nan_exception)

                if validate and (
                        np.mod(epoch - start_epoch, 5) == 0 or
                        epoch == n_epochs - 1):
                    valid_results = []
                    for i, (X_v, y_v) in enumerate(
                            valid_batch_provider.iter_pieces()):
                        valid_results.append(self.valid_fun(X_v, y_v))

                    valid_loss = np.nanmean(valid_results, axis=0)

                    write_append(
                        valid_loss_fn, epoch, valid_loss[0])
                    write_append(
                        valid_loss_2_fn, epoch, valid_loss[-1])

                named_valid_results = zip([o.variable.name for o in
                                           self.valid_fun.outputs],
                                          valid_loss)
                LOGGER.info(
                    ("Epoch: {0}/{3}, "
                     "train: {1}, "
                     "validate: {2} ")
                    .format(epoch,
                            '; '.join(
                                '{0} ={1: .6e}'.format(k, v) for k, v in
                                named_train_results),
                            '; '.join(
                                '{0} ={1: .6e}'.format(k, v) for k, v in
                                named_valid_results),
                            n_epochs))

                params = self.get_params()

                # Early stopping
                if validate:
                    es_loss = valid_loss[0]

                else:
                    es_loss = train_loss[0]

                if es_loss < best_loss:
                    best_params = params
                    best_loss = es_loss
                    best_epoch = epoch
                # Make a backup every 100 epochs (Astrud is sometimes
                # unreliable)
                if np.mod(epoch - start_epoch, 100) == 0:
                    LOGGER.info('Backing parameters up!')
                    save_pyc_bz(best_params,
                                os.path.join(self.out_dir, 'backup_params.pyc.bz'))

                early_stop = (
                    epoch + 1 > (best_epoch + max_epochs_from_best))

                if early_stop:
                    break

        except (RuntimeError, KeyboardInterrupt) as e:
            print('Training interrupted')

        if best_loss < np.inf:
            print('Reloading best self (epoch = {0}, {2} loss = {1:.3f})'
                  .format(best_epoch + 1, best_loss,
                          'validation' if validate else 'training'))

            self.set_params(best_params)

        return self.get_params()


def _train_loss(model, batch_provider, batch_size, seq_length,
                X_t, y_t, train_loss_fn, epoch,
                batches_per_epoch, nan_exception, mode='valid'):

    train_results = []

    if y_t is not None:
        inputs = [X_t, y_t]
    else:
        inputs = [X_t]

    # Select training function
    if mode == 'valid':
        get_batch = batch_provider.get_batch_valid
        if isinstance(model.train_fun, (list, tuple)):
            train_fun = model.train_fun[0]
        else:
            train_fun = model.train_fun
    elif mode == 'init':
        get_batch = batch_provider.get_batch_start
        if isinstance(model.train_fun, (list, tuple)):
            train_fun = model.train_fun[1]
        else:
            train_fun = model.train_fun
    elif mode == 'end':
        get_batch = batch_provider.get_batch_end
        if isinstance(model.train_fun, (list, tuple)):
            train_fun = model.train_fun[2]
        else:
            train_fun = model.train_fun

    elif mode == 'full':
        get_batch = batch_provider.get_batch_full
        if isinstance(model.train_fun, (list, tuple)):
            train_fun = model.train_fun[3]
        else:
            train_fun = model.train_fun

    for i in range(batches_per_epoch):
        # import pdb
        # pdb.set_trace()
        get_batch(*([batch_size, seq_length] + inputs))
        train_results.append(train_fun(*inputs[0]))

    train_loss = np.mean(train_results, axis=0)
    if any(np.isnan(train_loss)):
        if nan_exception:
            raise Exception("nan in training error")
        else:
            raise RuntimeError("nan in training, breaking training")
        # LOGGER.debug('Warning! NaN in loss. '
        # 'The following results cannot be trusted!')
        # # Replace NaNs with an absurdly large number (to know that
        # # something went horribly wrong)
        # train_loss[np.where(np.isnan(train_loss))] = np.finfo(np.float32).max

    write_append(train_loss_fn, epoch, train_loss[0])
    named_train_results = zip([o.variable.name for o in
                               train_fun.outputs],
                              train_loss)

    return train_loss, named_train_results


class SimpleParameterUpdate(object):

    def select_mode(self, epoch, crit=None):
        return 'valid'


class ParameterUpdate(object):

    def __init__(self, start_epoch, n_epochs):

        self.start_epoch = start_epoch
        self.n_epochs = n_epochs
        self.modes = cycle(['valid', 'init', 'end'])
        self.crit = 200
        self.crits = cycle([200, 25, 25])

    def select_mode(self, epoch, crit=None):

        # epochs = n_epochs - start_epoch
        if crit is None:
            crit = self.crit
        if np.mod(epoch - self.start_epoch, crit) == 0:
            self.mode = next(self.modes)
            self.crit = next(self.crits)

        return self.mode
