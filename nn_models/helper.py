"""
Helper methods for constructing neural networks
"""
import numpy as np
import theano.tensor as T
import theano
import lasagne


from lasagne import init
from lasagne import nonlinearities
from lasagne.layers import get_all_layers
from lasagne.layers import (
    NonlinearityLayer, BiasLayer,
    DropoutLayer, GaussianNoiseLayer,
    InputLayer, InverseLayer)

from custom_layers import MILSTMLayer
from melody_extractor.misc_tools import myMask

floatX = theano.config.floatX

# Set random state for all random initializations in Lasagne
lasagne.random.set_rng(np.random.RandomState(1984))


def get_rcnn_kwargs(input_width, input_height,
                    num_channels,
                    conv_num_filters,
                    conv_filter_size,
                    pool_size, rec_num_units,
                    p_dropout=0.5,
                    learning_rate=1e-3,
                    recurrent_layer_type=MILSTMLayer,
                    **kwargs):

    # Select number of gradient steps to be used for BPTT
    if 'gradient_steps' not in kwargs:
        gradient_steps = -1
    else:
        gradient_steps = kwargs['gradient_steps']

    # Input shape is (batch_size, sequence_length, num_channels,
    # input_height, input_width)
    input_shape = (None, None, num_channels, input_height, input_width,)

    # Define input layer
    l_in = lasagne.layers.InputLayer(input_shape)
    # symbolic variable for the input of the layer
    input_var = l_in.input_var
    # symbolic variables representing the size of the layer
    bsize, seqlen, _, _, _ = input_var.shape

    # Reshape layer for convolutional stage
    l_reshape_1 = lasagne.layers.ReshapeLayer(
        l_in, (-1, num_channels, input_height, input_width))

    # Convolutional stage
    l_conv = lasagne.layers.Conv2DLayer(l_reshape_1, conv_num_filters,
                                        conv_filter_size,
                                        nonlinearity=None, b=None)
    # Add bias and nonlinearities (so that they don't get included
    # in the computation of the gradient in the inverse layer
    # np.random.seed(84)
    # bias = np.random.rand(conv_num_filters).astype(floatX) % 0.1
    l_conv_b = lasagne.layers.BiasLayer(l_conv)  # , b=bias)
    l_conv_nl = lasagne.layers.NonlinearityLayer(l_conv_b,
                                                 nonlinearity=nonlinearities.sigmoid)
    # Max pooling layer
    l_pool = lasagne.layers.MaxPool2DLayer(l_conv_nl,
                                           pool_size=pool_size)

    # Dropout layer
    l_dropout = lasagne.layers.DropoutLayer(l_pool,
                                            p_dropout)

    # output shape for the convolutional stage
    out_shape = l_pool.output_shape

    # Reshape layer for recurrent stage
    # the shape is now (batch_size, sequence_length, num_channels * input_width *
    # input_height)
    l_reshape_2 = lasagne.layers.ReshapeLayer(
        l_dropout, (bsize, seqlen, np.prod(out_shape[1:])))

    # Recurrent stage
    l_rec = recurrent_layer_type(
        l_reshape_2, rec_num_units)

    # Decoding stage
    l_rec_d = recurrent_layer_type(
        l_rec, num_units=l_rec.input_shapes[0][-1])

    # Reshape output for decoding convolutional layers
    l_reshape_3 = lasagne.layers.ReshapeLayer(
        l_rec_d, (bsize * seqlen, ) + out_shape[1:])

    # Invert pool layer
    l_inv_pool = lasagne.layers.InverseLayer(l_reshape_3, l_pool)

    # Invert convolutional layers
    l_inv_conv = lasagne.layers.InverseLayer(l_inv_pool, l_conv)
    # Add nonlinearity
    # l_inv_conv_nl = l_inv_conv
    l_inv_conv_nl = lasagne.layers.NonlinearityLayer(l_inv_conv,
                                                     nonlinearity=nonlinearities.sigmoid)

    # Output layer (with same shape as the input)
    l_out = lasagne.layers.ReshapeLayer(
        l_inv_conv_nl, (bsize, seqlen, num_channels, input_height, input_width))

    target = T.tensor5()

    # Mask output
    # Are binary masks necessary?
    deterministic_out = myMask(
        lasagne.layers.get_output(l_out, deterministic=True),
        input_var)
    stochastic_out = myMask(lasagne.layers.get_output(l_out),
                            input_var)

    # Get parameters
    params = lasagne.layers.get_all_params(l_out, trainable=True)

    # Compute training and validation losses
    stochastic_loss = lasagne.objectives.binary_crossentropy(
        stochastic_out, target).mean()
    stochastic_loss.name = 'stochastic loss'
    deterministic_loss = lasagne.objectives.binary_crossentropy(
        deterministic_out, target).mean()
    deterministic_loss.name = 'deterministic loss'

    # Compute updates
    updates = lasagne.updates.rmsprop(
        stochastic_loss, params, learning_rate=learning_rate)

    train_loss = [stochastic_loss]
    valid_loss = [deterministic_loss]

    return dict(l_in=l_in, l_out=l_out,
                train_loss=train_loss,
                valid_loss=valid_loss,
                # input_var=input_var,
                target=target, updates=updates,
                predictions=deterministic_out,
                gradient_steps=gradient_steps,
                model_type='RNN')


# if __name__ == '__main__':

#     from models import RNN

#     width, height = (40, 30)
#     num_channels = 1
#     is_recurrent = True
#     num_filters = 2
#     filter_size = (3, 3)
#     pool_size = (2, 2)
#     num_units = 3

#     model_kwargs = convolutional_recurrent_network(width, height,
#                                                    num_channels,
#                                                    num_filters,
#                                                    filter_size,
#                                                    pool_size,
#                                                    num_units,
#                                                    gradient_steps=5)  # Gradient steps is important!

#     model_kwargs.pop('model_type')
#     X = np.random.rand(100, 10, num_channels, width, height).astype(floatX)

#     rnn = RNN(**model_kwargs)

#     # Test that it works
#     rnn.fit(X, X, X, X)
