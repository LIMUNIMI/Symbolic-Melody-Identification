import bz2
import logging
import math
import os
import time

import lasagne
import numpy as np
import theano
import theano.tensor as T

from melody_extractor import settings, misc_tools


# from utils import (delete_if_exists,
#                    write_append)
try:
    import cPickle
except Exception:
    import pickle

floatX = settings.floatX
logging.basicConfig()
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
logging.getLogger('visualize').setLevel(logging.INFO)


def save_pyc_bz(d, fn):
    # Copied from extra.utils.os_utils
    cPickle.dump(d, bz2.BZ2File(fn, 'w'), cPickle.HIGHEST_PROTOCOL)


class ModifiedBackprop(object):
    """
    A class to modify backpropagation to the end of saliency maps.
    See: https://github.com/Lasagne/Recipes/blob/master/examples/Saliency%20Maps%20and%20Guided%20Backpropagation.ipynb
    """

    def __init__(self, nonlinearity):
        self.nonlinearity = nonlinearity
        self.ops = {}  # memoizes an OpFromGraph instance per tensor type

    def __call__(self, x):
        # OpFromGraph is oblique to Theano optimizations, so we need to move
        # things to GPU ourselves if needed.
        if theano.sandbox.cuda.cuda_enabled:
            maybe_to_gpu = theano.sandbox.cuda.as_cuda_ndarray_variable
        else:
            def maybe_to_gpu(x): return x
        # We move the input to GPU if needed.
        x = maybe_to_gpu(x)
        # We note the tensor type of the input variable to the nonlinearity
        # (mainly dimensionality and dtype); we need to create a fitting Op.
        tensor_type = x.type
        # If we did not create a suitable Op yet, this is the time to do so.
        if tensor_type not in self.ops:
            # For the graph, we create an input variable of the correct type:
            inp = tensor_type()
            # We pass it through the nonlinearity (and move to GPU if needed).
            outp = maybe_to_gpu(self.nonlinearity(inp))
            # Then we fix the forward expression...
            op = theano.OpFromGraph([inp], [outp])
            # ...and replace the gradient with our own (defined in a subclass).
            op.grad = self.grad
            # Finally, we memoize the new Op
            self.ops[tensor_type] = op
        # And apply the memoized Op to the input we got.
        return self.ops[tensor_type](x)


class GuidedBackprop(ModifiedBackprop):
    """
    Guided backpropagation: https://github.com/Lasagne/Recipes/blob/master/examples/Saliency%20Maps%20and%20Guided%20Backpropagation.ipynb
    """

    def grad(self, inputs, out_grads):
        (inp,) = inputs
        (grd,) = out_grads
        dtype = inp.dtype
        return (grd * (inp > 0).astype(dtype) * (grd > 0).astype(dtype),)


def iterate_minibatches(arr, batchsize):
    """
    iterator on arr. At each iteration returns an excerpt of arr of size
    *batchsize*
    """
    start_idx = 0
    for start_idx in range(0, len(arr) - batchsize + 1, batchsize):
        excerpt = slice(start_idx, start_idx + batchsize)
        yield arr[excerpt]

    # put remaining data in a last mini-batch
    start_idx += batchsize
    if start_idx > len(arr):
        excerpt = slice(start_idx, len(arr))
        yield arr[excerpt]


class CNN(object):
    """
    Base class for training convolutional neural networks, with same size in
    input and output.


    This particular implementation uses *n_layer* as the number of
    convolutional layers. For each convolutional layer, another deconvolutional
    layer is created.

    *input_l* is the input of this network
    *l_in* is the layer with which the output functions and the masking are
    computed. Use two different layers just if you are connecting together
    different networks
    """

    def __init__(self, input_l, l_in,
                 train_loss_fn_name,
                 valid_loss_fn_name,
                 updates_fn_name,
                 target,
                 num_kernel,
                 kernel_h,
                 kernel_w,
                 num_layers,
                 nonlinearity,
                 dropout_p,
                 LEARNING_RATE,
                 initializer,
                 out_dir='/tmp',
                 MASKING=True,
                 TYPE='output'):

        self.switch = False
        self.l_in = input_l
        self.nonlinearity = nonlinearity

        # Set path to save output files
        self.out_dir = out_dir
        np.random.seed(42)
        bias = None  # np.random.rand(num_kernel).astype(floatX) % 0.1

        l_conv1 = lasagne.layers.Conv2DLayer(
            input_l, num_kernel, (kernel_h, kernel_w), b=bias,
            nonlinearity=None, W=initializer(),  # )
            stride=1
        )
        lasagne.layers.batch_norm(l_conv1)
        l_dropout = lasagne.layers.DropoutLayer(
            l_conv1, dropout_p, rescale=True)
        print("Conv-bn-dropout size out: " + str(l_dropout.output_shape))
        l_conv2 = lasagne.layers.Conv2DLayer(
            l_conv1, num_kernel, (kernel_h, kernel_w), b=bias,
            nonlinearity=nonlinearity, W=initializer(),  # )
            stride=1
        )
        print("Convolutional 2 size out: " + str(l_conv2.output_shape))

        my_layers = [l_conv1, l_conv2]

        incoming_l = my_layers[-1]
        for i in range(len(my_layers)):
            incoming_l = lasagne.layers.InverseLayer(
                incoming_l, my_layers[-i - 1])
            print("inverse layer size out: " + str(incoming_l.output_shape))
        incoming_l = lasagne.layers.NonlinearityLayer(
            incoming_l, nonlinearity)

        # apply regularization to the whole network
        lasagne.regularization.regularize_network_params(
            incoming_l, lasagne.regularization.l1)

        self.l_out = incoming_l
        self.train_output = lasagne.layers.get_output(
            incoming_l, deterministic=False)

        self.valid_output = lasagne.layers.get_output(
            incoming_l, deterministic=True)

        if MASKING:
            # masking
            self.predict_output = l_in.input_var * self.valid_output
            self.train_output_masked = l_in.input_var * self.train_output
            self.valid_output_masked = l_in.input_var * self.valid_output

        else:
            self.predict_output = self.valid_output

        if TYPE == 'output':
            params = lasagne.layers.get_all_params(incoming_l)

            train_loss = T.sqrt(train_loss_fn_name(
                self.train_output, target).sum() / l_in.input_var.sum())
            train_loss_masked = T.sqrt(train_loss_fn_name(
                self.train_output_masked, target).sum() / l_in.input_var.sum())

            valid_loss = T.sqrt(valid_loss_fn_name(
                self.valid_output, target).sum() / l_in.input_var.sum())
            valid_loss_masked = T.sqrt(valid_loss_fn_name(
                self.valid_output_masked, target).sum() / l_in.input_var.sum())

            updates = updates_fn_name(
                train_loss, params, learning_rate=LEARNING_RATE)
            updates_masked = updates_fn_name(
                train_loss_masked, params, learning_rate=LEARNING_RATE)

            # y_fun = theano.function([l_in.input_var], y)
            self.train_fn = theano.function(
                [l_in.input_var, target], train_loss, updates=updates)
            self.train_fn_masked = theano.function(
                [l_in.input_var, target], train_loss_masked, updates=updates_masked)

            self.val_fn = theano.function(
                [l_in.input_var, target], valid_loss)
            self.val_fn_masked = theano.function(
                [l_in.input_var, target], valid_loss_masked)

            self.output = theano.function(
                [l_in.input_var], self.predict_output)

            self.saliency = self.compile_saliency_function()

    def fit(self, X, Y, tr_map, val_map=None,
            NUM_EPOCHS=100, BATCHSIZE=10,
            max_epochs_from_best=20,
            keep_training=False,
            nan_exception=False,
            masked=False):

        start_epoch = 0
        best_epoch = 0
        best_loss = np.inf
        best_params = self.get_params()
        validate = val_map is not None
        firstRun = True

        def epoch_train():
            # In each epoch, we do a full pass over the training data:
            train_err = 0
            train_samples = 0
            start_time = time.time()

            for batch in iterate_minibatches(tr_map, BATCHSIZE):
                # batch are the indices of the windows
                inputs = X[batch, :, :, :]
                targets = Y[batch, :, :, :]
                if self.switch:
                    targets = inputs - targets

                if masked:
                    train_err += self.train_fn_masked(inputs, targets)
                    train_samples += 1  # inputs[inputs == 1].size
                else:
                    train_err += self.train_fn(inputs, targets)
                    train_samples += inputs.size

            if np.isnan(train_err).any():
                if nan_exception or epoch == 0:
                    raise Exception("nan in training error")
                else:
                    print("Found nan!")
                    self.set_params(best_params)
                    raise RuntimeError(
                        "nan in training, breaking training")

            # And a full pass over the validation data:
            if validate:
                val_err = 0
                val_samples = 0
                for batch in iterate_minibatches(val_map, BATCHSIZE):
                    inputs = X[batch, :, :, :]
                    targets = Y[batch, :, :, :]
                    if self.switch:
                        targets = inputs - targets

                    if masked:
                        val_err += self.val_fn_masked(inputs, targets)
                        val_samples += 1  # inputs[inputs == 1].size
                    else:
                        val_err += self.val_fn(inputs, targets)
                        val_samples += inputs.size
            else:
                val_err = 0
                val_samples = 1

            epoch_time = time.time() - start_time
            train_loss = train_err / train_samples
            val_loss = val_err / val_samples
            LOGGER.info(
                (
                    "Epoch {} of {} took {:.3f}s"
                    "  training loss:\t{:.6f}"
                    "  validation loss:\t{:.6f}"
                ).format(
                    epoch + 1, NUM_EPOCHS, epoch_time,
                    train_loss, val_loss)
            )

            return train_loss, val_loss, epoch_time

        try:
            for epoch in xrange(start_epoch,
                                NUM_EPOCHS):

                if epoch == 0:
                    # let's try if it's better switching or not
                    LOGGER.info("Starting epoch without switching..")
                    self.switch = False
                    switch_false_train_loss, switch_false_val_loss, epoch_time = epoch_train()
                    switch_false_params = self.get_params()
                    self.set_params(best_params)
                    LOGGER.info("Starting epoch with switching..")
                    self.switch = True
                    switch_true_train_loss, switch_true_val_loss, _epoch_time = epoch_train()
                    epoch_time = (epoch_time + _epoch_time) / 2
                    if switch_true_val_loss < switch_false_val_loss:
                        LOGGER.info("Using switching...")
                        train_loss = switch_true_train_loss
                        val_loss = switch_true_val_loss
                        self.switch = True
                    else:
                        LOGGER.info("Not using switching")
                        train_loss = switch_false_train_loss
                        val_loss = switch_false_val_loss
                        self.set_params(switch_false_params)
                        self.switch = False
                else:
                    train_loss, val_loss, epoch_time = epoch_train()

                params = self.get_params()

                # Early stopping
                if validate:
                    es_loss = val_loss

                else:
                    es_loss = train_loss

                if es_loss < best_loss:
                    best_params = params
                    best_loss = es_loss
                    best_epoch = epoch

                early_stop = (
                    epoch > (best_epoch + max_epochs_from_best))

                if epoch_time > settings.MAX_TIME:
                    raise RuntimeError(
                        'too much time needed for an epoch, breaking training')
                elif early_stop:
                    break

        except (RuntimeError, KeyboardInterrupt) as e:
            print('Training interrupted: ' + str(e))

        if best_loss < np.inf:
            print('Reloading best self (epoch = {0}, {2} loss = {1:.3f})'
                  .format(best_epoch + 1, best_loss,
                          'validation' if validate else 'training'))

            self.set_params(best_params)

        return self.get_params()

    def get_params(self):
        """
        Get the parameters of the network.

        Returns
        -------
        iterable of arrays
        A list with the values of all parameters that specify the
        neural network. Last parameter must be the 'switch' field.
        """
        params = lasagne.layers.get_all_param_values(self.l_out)
        params.append(self.switch)
        return params

    def set_params(self, params):
        """
        Set the parameters of the neural network

        Parameters
        ----------
        params : iterable of arrays
        An iterable that contains the values of all parameters that
        specify the neural network. Each parameters must be of the
        same size and dtype as the original parameters. Last parameter must be
        the 'switch' field
        """
        self.switch = params[-1]
        return lasagne.layers.set_all_param_values(self.l_out, params[:-1])

    def compile_saliency_function(self):
        """
        Compiles a function to compute the saliency maps and predicted classes
        for a given minibatch of input images. Again, see:
            https://github.com/Lasagne/Recipes/blob/master/examples/Saliency%20Maps%20and%20Guided%20Backpropagation.ipynb
        """
        inp = self.l_in.input_var
        outp = self.predict_output
        saliency = theano.grad(outp.sum(), wrt=inp)[0]
        return theano.function([inp], saliency[:, np.newaxis, :, :])

    def insert_guided_backprop(self):
        """
        Change all nonlinearities with the guided backpropagation. Use this
        to compute saliency maps
        Again, again, see: https://github.com/Lasagne/Recipes/blob/master/examples/Saliency%20Maps%20and%20Guided%20Backpropagation.ipynb
        """
        nonlinear_layers = [layer for layer in lasagne.layers.get_all_layers(self.l_out)
                            if getattr(layer, 'nonlinearity', None) is self.nonlinearity]
        # important: only instantiate this once!
        modded_nonlin = GuidedBackprop(self.nonlinearity)
        for layer in nonlinear_layers:
            layer.nonlinearity = modded_nonlin

    def guided_saliency(self, X):
        self.insert_guided_backprop()
        return self.saliency(X)

    def predict(self, X):
        """
        Compute predictions of the neural network for the given input.

        Parameters
        ----------
        X : array
        4D input array for the neural network.

        Returns
        -------
        array
        Predictions of the neural network.
        """

        if self.switch:
            return X - self.output(X)
        else:
            return self.output(X)
