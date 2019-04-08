# -*- coding: utf-8 -*-
"""
Layers to construct recurrent networks with multiplicative connections.
"""
import numpy as np
import theano
import theano.tensor as T

from lasagne.layers import (Layer,
                            InputLayer,
                            DenseLayer,
                            CustomRecurrentLayer,
                            MergeLayer,
                            helper,
                            BatchNormLayer)
from lasagne import (nonlinearities,
                     init)

from lasagne.utils import unroll_scan

__all__ = [
    "CustomMIRecurrentLayer",
    "MIRecurrentLayer",
    "MIGate",
    "MILSTMLayer",
    "MIGRULayer"
]


class CustomMIRecurrentLayer(CustomRecurrentLayer):

    """
    A layer which implements a recurrent connection with multiplicative integration.
    """

    def __init__(self, incoming, input_to_hidden, hidden_to_hidden,
                 a_g=init.Uniform(0.1),
                 b_g_in_to_hid=init.Uniform(0.1),
                 b_g_hid_to_hid=init.Uniform(0.1),
                 b=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify,
                 hid_init=init.Constant(0.),
                 backwards=False,
                 learn_init=False,
                 learn_a_g=True,
                 learn_b_g_in_to_hid=True,
                 learn_b_g_hid_to_hid=True,
                 gradient_steps=-1,
                 grad_clipping=0,
                 unroll_scan=False,
                 precompute_input=True,
                 mask_input=None,
                 only_return_final=False,
                 **kwargs):

        super(CustomMIRecurrentLayer, self).__init__(
            incoming, input_to_hidden, hidden_to_hidden,
            nonlinearity=nonlinearity,
            hid_init=hid_init,
            backwards=backwards,
            learn_init=learn_init,
            gradient_steps=gradient_steps,
            grad_clipping=grad_clipping,
            unroll_scan=unroll_scan,
            precompute_input=precompute_input,
            mask_input=mask_input,
            only_return_final=only_return_final,
            **kwargs)

        # Initialize second order gating bias
        if a_g is None:
            self.a_g = None
        else:
            self.a_g = self.add_param(
                a_g, hidden_to_hidden.output_shape[1:],
                name="a_g", trainable=learn_a_g, regularizable=False)

        # Initialize hidden to hidden gating bias
        if b_g_hid_to_hid is None:
            self.b_g_hid_to_hid = None
        else:
            self.b_g_hid_to_hid = self.add_param(
                b_g_hid_to_hid, hidden_to_hidden.output_shape[1:],
                name="b_g_hid_to_hid", trainable=learn_b_g_hid_to_hid,
                regularizable=False)

        # Initialize input to hidden gatig bias
        if b_g_in_to_hid is None:
            self.b_g_in_to_hid = None
        else:
            self.b_g_in_to_hid = self.add_param(
                b_g_in_to_hid, hidden_to_hidden.output_shape[1:],
                name="b_g_in_to_hid", trainable=learn_b_g_in_to_hid,
                regularizable=False)

        # Initialize additive bias
        if b is None:
            self.b = None
        else:
            self.b = self.add_param(
                b, hidden_to_hidden.output_shape[1:],
                name="b", trainable=True, regularizable=False)

    def get_params(self, **tags):
        # Get all parameters from this layer, the master layer
        params = super(CustomMIRecurrentLayer, self).get_params(**tags)
        # Combine with all parameters from the child layers
        params += helper.get_all_params(self.input_to_hidden, **tags)
        params += helper.get_all_params(self.hidden_to_hidden, **tags)
        return params

    def _get_mi_params(self):
        # Get parameters of the multiplicative integration
        params = []
        if self.a_g is not None:
            params.append(self.a_g)

        if self.b_g_in_to_hid is not None:
            params.append(self.b_g_in_to_hid)

        if self.b_g_hid_to_hid is not None:
            params.append(self.b_g_hid_to_hid)

        if self.b is not None:
            params.append(self.b)

        return params

    def get_output_for(self, inputs, **kwargs):
        """
        Compute this layer's output function given a symbolic input variable.

        Parameters
        ----------
        inputs : list of theano.TensorType
            `inputs[0]` should always be the symbolic input variable.  When
            this layer has a mask input (i.e. was instantiated with
            `mask_input != None`, indicating that the lengths of sequences in
            each batch vary), `inputs` should have length 2, where `inputs[1]`
            is the `mask`.  The `mask` should be supplied as a Theano variable
            denoting whether each time step in each sequence in the batch is
            part of the sequence or not.  `mask` should be a matrix of shape
            ``(n_batch, n_time_steps)`` where ``mask[i, j] = 1`` when ``j <=
            (length of sequence i)`` and ``mask[i, j] = 0`` when ``j > (length
            of sequence i)``. When the hidden state of this layer is to be
            pre-filled (i.e. was set to a :class:`Layer` instance) `inputs`
            should have length at least 2, and `inputs[-1]` is the hidden state
            to prefill with.

        Returns
        -------
        layer_output : theano.TensorType
            Symbolic output variable.
        """
        # Retrieve the layer input
        input = inputs[0]
        # Retrieve the mask when it is supplied
        mask = None
        hid_init = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]
        if self.hid_init_incoming_index > 0:
            hid_init = inputs[self.hid_init_incoming_index]

        # Input should be provided as (n_batch, n_time_steps, n_features)
        # but scan requires the iterable dimension to be first
        # So, we need to dimshuffle to (n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0, *range(2, input.ndim))
        seq_len, num_batch = input.shape[0], input.shape[1]

        if self.precompute_input:
            # Because the input is given for all time steps, we can precompute
            # the inputs to hidden before scanning. First we need to reshape
            # from (seq_len, batch_size, trailing dimensions...) to
            # (seq_len*batch_size, trailing dimensions...)
            # This strange use of a generator in a tuple was because
            # input.shape[2:] was raising a Theano error
            trailing_dims = tuple(input.shape[n] for n in range(2, input.ndim))
            input = T.reshape(input, (seq_len * num_batch,) + trailing_dims)
            input = helper.get_output(
                self.input_to_hidden, input, **kwargs)

            # Reshape back to (seq_len, batch_size, trailing dimensions...)
            trailing_dims = tuple(input.shape[n] for n in range(1, input.ndim))
            input = T.reshape(input, (seq_len, num_batch) + trailing_dims)

        # We will always pass the hidden-to-hidden layer params to step
        non_seqs = helper.get_all_params(self.hidden_to_hidden)
        non_seqs += self._get_mi_params()
        # When we are not precomputing the input, we also need to pass the
        # input-to-hidden parameters to step
        if not self.precompute_input:
            non_seqs += helper.get_all_params(self.input_to_hidden)

        # Create single recurrent computation step function
        def step(input_n, hid_previous, *args):
            # Compute the hidden-to-hidden activation
            hid_to_hid = helper.get_output(
                self.hidden_to_hidden, hid_previous, **kwargs)

            # Compute the input-to-hidden activation
            if self.precompute_input:
                # if the input is precomputed
                in_to_hid = input_n
            else:
                # compute the input
                in_to_hid = helper.get_output(
                    self.input_to_hidden, input_n, **kwargs)

            # Compute the second order term
            if self.a_g is not None:
                second_order_term = (self.a_g * in_to_hid *
                                     hid_to_hid)
                # second_order_term = in_to_hid * hid_to_hid
            else:
                second_order_term = 0

            # Compute the first order hidden-to-hidden term
            if self.b_g_hid_to_hid is not None:
                f_o_hid_to_hid = self.b_g_hid_to_hid * hid_to_hid

            else:
                f_o_hid_to_hid = 0

            # Compute first order input to hidden term
            if self.b_g_in_to_hid is not None:
                f_o_in_to_hid = self.b_g_in_to_hid * in_to_hid

            else:
                # if all else is None, it will output zeros of the right size
                f_o_in_to_hid = T.zeros_like(in_to_hid)

            hid_pre = second_order_term + f_o_in_to_hid + f_o_hid_to_hid

            if self.b is not None:
                hid_pre = hid_pre + self.b

            return self.nonlinearity(hid_pre)

        def step_masked(input_n, mask_n, hid_previous, *args):
            # Skip over any input with mask 0 by copying the previous
            # hidden state; proceed normally for any input with mask 1.
            hid = step(input_n, hid_previous, *args)
            hid_out = T.switch(mask_n, hid, hid_previous)
            return [hid_out]

        if mask is not None:
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, mask]
            step_fun = step_masked
        else:
            sequences = input
            step_fun = step

        if not isinstance(self.hid_init, Layer):
            # The code below simply repeats self.hid_init num_batch times in
            # its first dimension.  Turns out using a dot product and a
            # dimshuffle is faster than T.repeat.
            dot_dims = (list(range(1, self.hid_init.ndim - 1)) +
                        [0, self.hid_init.ndim - 1])
            hid_init = T.dot(T.ones((num_batch, 1)),
                             self.hid_init.dimshuffle(dot_dims))

        if self.unroll_scan:
            # Retrieve the dimensionality of the incoming layer
            input_shape = self.input_shapes[0]
            # Explicitly unroll the recurrence instead of using scan
            hid_out = unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[hid_init],
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=input_shape[1])[0]
        else:
            # Scan op iterates over first dimension of input and repeatedly
            # applies the step function
            hid_out = theano.scan(
                fn=step_fun,
                sequences=sequences,
                go_backwards=self.backwards,
                outputs_info=[hid_init],
                non_sequences=non_seqs,
                truncate_gradient=self.gradient_steps,
                strict=True)[0]

        # When it is requested that we only return the final sequence step,
        # we need to slice it out immediately after scan is applied
        if self.only_return_final:
            hid_out = hid_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            hid_out = hid_out.dimshuffle(1, 0, *range(2, hid_out.ndim))

            # if scan is backward reverse the output
            if self.backwards:
                hid_out = hid_out[:, ::-1]

        return hid_out


class MIRecurrentLayer(CustomMIRecurrentLayer):

    def __init__(self, incoming, num_units,
                 W_in_to_hid=init.Uniform(),
                 W_hid_to_hid=init.Uniform(),
                 a_g=init.Uniform(0.1),
                 b_g_hid_to_hid=init.Uniform(0.1),
                 b_g_in_to_hid=init.Uniform(0.1),
                 b=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify,
                 hid_init=init.Constant(0.),
                 backwards=False,
                 learn_init=False,
                 learn_a_g=True,
                 learn_b_g_in_to_hid=True,
                 learn_b_g_hid_to_hid=True,
                 gradient_steps=-1,
                 grad_clipping=0,
                 unroll_scan=False,
                 precompute_input=True,
                 mask_input=None,
                 only_return_final=False,
                 **kwargs):

        if isinstance(incoming, tuple):
            input_shape = incoming
        else:
            input_shape = incoming.output_shape
        # Retrieve the supplied name, if it exists; otherwise use ''
        if 'name' in kwargs:
            basename = kwargs['name'] + '.'
            # Create a separate version of kwargs for the contained layers
            # which does not include 'name'
            layer_kwargs = dict((key, arg) for key, arg in kwargs.items()
                                if key != 'name')
        else:
            basename = ''
            layer_kwargs = kwargs
        # We will be passing the input at each time step to the dense layer,
        # so we need to remove the second dimension (the time dimension)
        in_to_hid = DenseLayer(InputLayer((None,) + input_shape[2:]),
                               num_units, W=W_in_to_hid, b=None,
                               nonlinearity=None,
                               name=basename + 'input_to_hidden',
                               **layer_kwargs)
        # The hidden-to-hidden layer expects its inputs to have num_units
        # features because it recycles the previous hidden state
        hid_to_hid = DenseLayer(InputLayer((None, num_units)),
                                num_units, W=W_hid_to_hid, b=None,
                                nonlinearity=None,
                                name=basename + 'hidden_to_hidden',
                                **layer_kwargs)

        # Make child layer parameters intuitively accessible
        self.W_in_to_hid = in_to_hid.W
        self.W_hid_to_hid = hid_to_hid.W

        super(MIRecurrentLayer, self).__init__(
            incoming, in_to_hid, hid_to_hid,
            a_g=a_g,
            b_g_in_to_hid=b_g_in_to_hid,
            b_g_hid_to_hid=b_g_hid_to_hid,
            b=b,
            nonlinearity=nonlinearity,
            hid_init=hid_init,
            backwards=backwards,
            learn_init=learn_init,
            learn_a_g=learn_a_g,
            learn_b_g_in_to_hid=learn_b_g_in_to_hid,
            gradient_steps=gradient_steps,
            grad_clipping=grad_clipping,
            unroll_scan=unroll_scan,
            precompute_input=precompute_input,
            mask_input=mask_input,
            only_return_final=only_return_final,
            **kwargs)


class MIGate(object):

    def __init__(self, W_in=init.Normal(0.1),
                 W_hid=init.Normal(0.1),
                 # W_cell=init.Normal(0.1),
                 b=init.Constant(0.),
                 a_g=init.Uniform(0.1),
                 b_g_hid_to_hid=init.Uniform(0.1),
                 b_g_in_to_hid=init.Uniform(0.1),
                 nonlinearity=nonlinearities.sigmoid,
                 learn_a_g=True,
                 learn_b_g_in_to_hid=True,
                 learn_b_g_hid_to_hid=True
                 ):
        # TODO: Make formulation with peepholes and W_cell
        self.W_in = W_in
        self.W_hid = W_hid
        # if W_cell is not None:
        #     self.W_cell = W_cell
        self.a_g = a_g
        if a_g is not None:
            self.learn_a_g = learn_a_g
        self.b_g_in_to_hid = b_g_in_to_hid
        if b_g_hid_to_hid is not None:
            self.learn_b_g_in_to_hid = learn_b_g_hid_to_hid
        self.b_g_hid_to_hid = b_g_hid_to_hid
        if b_g_hid_to_hid is not None:
            self.learn_b_g_hid_to_hid = learn_b_g_hid_to_hid
        self.b = b
        # For the nonlinearity, if None is supplied, use identity
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity


class MILSTMLayer(MergeLayer):
    # TODO: add formulation with peepholes

    def __init__(self, incoming, num_units,
                 ingate=MIGate(),
                 forgetgate=MIGate(),
                 cell=MIGate(nonlinearity=nonlinearities.tanh),
                 outgate=MIGate(),
                 nonlinearity=nonlinearities.tanh,
                 cell_init=init.Constant(0.),
                 hid_init=init.Constant(0.),
                 backwards=False,
                 learn_init=False,
                 gradient_steps=-1,
                 grad_clipping=0,
                 unroll_scan=False,
                 precompute_input=True,
                 mask_input=None,
                 only_return_final=False,
                 bn=False,
                 **kwargs):

        incomings = [incoming]
        self.mask_incoming_index = -1
        self.hid_init_incoming_index = -1
        self.cell_init_incoming_index = -1
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = len(incomings) - 1
        if isinstance(hid_init, Layer):
            incomings.append(hid_init)
            self.hid_init_incoming_index = len(incomings) - 1
        if isinstance(cell_init, Layer):
            incomings.append(cell_init)
            self.cell_init_incoming_index = len(incomings) - 1

        # Initialize parent layer
        super(MILSTMLayer, self).__init__(incomings, **kwargs)

        # If the provided nonlinearity is None, make it linear
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        self.learn_init = learn_init
        self.num_units = num_units
        self.backwards = backwards
        self.gradient_steps = gradient_steps
        self.grad_clipping = grad_clipping
        self.unroll_scan = unroll_scan
        self.precompute_input = precompute_input
        self.only_return_final = only_return_final

        if unroll_scan and gradient_steps != -1:
            raise ValueError(
                "Gradient steps must be -1 when unroll_scan is true.")

        # Retrieve the dimensionality of the incoming layer
        input_shape = self.input_shapes[0]

        if unroll_scan and input_shape[1] is None:
            raise ValueError("Input sequence length cannot be specified as "
                             "None when unroll_scan is True")

        num_inputs = np.prod(input_shape[2:])

        def add_gate_params(gate, gate_name):
            """ Convenience function for adding layer parameters from a Gate
            instance. """

            W_in = self.add_param(gate.W_in, (num_inputs, num_units),
                                  name="W_in_to_{}".format(gate_name))
            W_hid = self.add_param(gate.W_hid, (num_units, num_units),
                                   name="W_hid_to_{}".format(gate_name))
            b = self.add_param(gate.b, (num_units,),
                               name="b_{}".format(gate_name),
                               regularizable=False)
            if gate.a_g is not None:
                a_g = self.add_param(gate.a_g, (num_units,),
                                     name="a_g_to_{}".format(gate_name),
                                     regularizable=False,
                                     trainable=gate.learn_a_g)
            else:
                a_g = None

            if gate.b_g_in_to_hid is not None:
                b_g_in_to_hid = self.add_param(
                    gate.b_g_in_to_hid, (num_units,),
                    name="b_g_in_to_hid_to_{}".format(gate_name),
                    regularizable=False,
                    trainable=gate.learn_b_g_in_to_hid)
            else:
                b_g_in_to_hid = None

            if gate.b_g_hid_to_hid is not None:
                b_g_hid_to_hid = self.add_param(
                    gate.b_g_hid_to_hid, (num_units, ),
                    name="b_g_hid_to_hid_to_{}".format(gate_name),
                    regularizable=False,
                    trainable=gate.learn_b_g_hid_to_hid)

            return (W_in, W_hid, b, a_g, b_g_in_to_hid, b_g_hid_to_hid,
                    gate.nonlinearity)

        # Add in parameters from the supplied Gate instances
        (self.W_in_to_ingate, self.W_hid_to_ingate, self.b_ingate,
         self.a_g_ingate, self.b_g_in_to_hid_ingate,
         self.b_g_hid_to_hid_ingate,
         self.nonlinearity_ingate) = add_gate_params(ingate, 'ingate')

        (self.W_in_to_forgetgate, self.W_hid_to_forgetgate, self.b_forgetgate,
         self.a_g_forgetgate, self.b_g_in_to_hid_forgetgate,
         self.b_g_hid_to_hid_forgetgate,
         self.nonlinearity_forgetgate) = add_gate_params(forgetgate,
                                                         'forgetgate')
        (self.W_in_to_cell, self.W_hid_to_cell, self.b_cell,
         self.a_g_cell, self.b_g_in_to_hid_cell,
         self.b_g_hid_to_hid_cell,
         self.nonlinearity_cell) = add_gate_params(cell, 'cell')

        (self.W_in_to_outgate, self.W_hid_to_outgate, self.b_outgate,
         self.a_g_outgate, self.b_g_in_to_hid_outgate,
         self.b_g_hid_to_hid_outgate,
         self.nonlinearity_outgate) = add_gate_params(outgate, 'outgate')

        # Setup initial values for the cell and the hidden units
        if isinstance(cell_init, Layer):
            self.cell_init = cell_init
        else:
            self.cell_init = self.add_param(
                cell_init, (1, num_units), name="cell_init",
                trainable=learn_init, regularizable=False)

        if isinstance(hid_init, Layer):
            self.hid_init = hid_init
        else:
            self.hid_init = self.add_param(
                hid_init, (1, self.num_units), name="hid_init",
                trainable=learn_init, regularizable=False)

        if bn:
            bn_shape = list(input_shape)
            bn_shape[-1] = 4 * self.num_units
            # create BN layer for correct input shape
            self.bn = BatchNormLayer(tuple(bn_shape), axes=(0, 1))
            self.params.update(self.bn.params)  # make BN params your params
        else:
            self.bn = False

    def get_output_shape_for(self, input_shapes):
        # The shape of the input to this layer will be the first element
        # of input_shapes, whether or not a mask input is being used.
        input_shape = input_shapes[0]
        # When only_return_final is true, the second (sequence step) dimension
        # will be flattened
        if self.only_return_final:
            return input_shape[0], self.num_units
        # Otherwise, the shape will be (n_batch, n_steps, num_units)
        else:
            return input_shape[0], input_shape[1], self.num_units

    def get_output_for(self, inputs, **kwargs):
        # Retrieve the layer input
        input = inputs[0]
        # Retrieve the mask when it is supplied
        mask = None
        hid_init = None
        cell_init = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]
        if self.hid_init_incoming_index > 0:
            hid_init = inputs[self.hid_init_incoming_index]
        if self.cell_init_incoming_index > 0:
            cell_init = inputs[self.cell_init_incoming_index]

        # Treat all dimensions after the second as flattened feature dimensions
        if input.ndim > 3:
            input = T.flatten(input, 3)

        # if self.bn:
        #     input = self.bn.get_output_for(input)

        # Because scan iterates over the first dimension we dimshuffle to
        # (n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0, 2)
        seq_len, num_batch, _ = input.shape

        # Stack input weight matrices into a (num_inputs, 4*num_units)
        # matrix, which speeds up computation
        W_in_stacked = T.concatenate(
            [self.W_in_to_ingate, self.W_in_to_forgetgate,
             self.W_in_to_cell, self.W_in_to_outgate], axis=1)

        # Same for hidden weight matrices
        W_hid_stacked = T.concatenate(
            [self.W_hid_to_ingate, self.W_hid_to_forgetgate,
             self.W_hid_to_cell, self.W_hid_to_outgate], axis=1)

        # Stack biases into a (4*num_units) vector
        b_stacked = T.concatenate(
            [self.b_ingate, self.b_forgetgate,
             self.b_cell, self.b_outgate], axis=0)

        # Stack second order term gating biases into a (4*num_units) vector
        a_g_stacked = T.concatenate(
            [self.a_g_ingate, self.a_g_forgetgate,
             self.a_g_cell, self.a_g_outgate], axis=0)

        # Stack second order term gating biases into a (4*num_units) vector
        b_g_in_to_hid_stacked = T.concatenate(
            [self.b_g_in_to_hid_ingate, self.b_g_in_to_hid_forgetgate,
             self.b_g_in_to_hid_cell, self.b_g_in_to_hid_outgate], axis=0)

        # Stack second order term gating biases into a (4*num_units) vector
        b_g_hid_to_hid_stacked = T.concatenate(
            [self.b_g_hid_to_hid_ingate, self.b_g_hid_to_hid_forgetgate,
             self.b_g_hid_to_hid_cell, self.b_g_hid_to_hid_outgate], axis=0)

        if self.precompute_input:
            # Because the input is given for all time steps, we can
            # precompute_input the inputs dot weight matrices before scanning.
            # W_in_stacked is (n_features, 4*num_units). input is then
            # (n_time_steps, n_batch, 4*num_units).
            input = T.dot(input, W_in_stacked)

            if self.bn:
                print 'Using batch normalization'
                input = self.bn.get_output_for(input.dimshuffle(1, 0, 2))
                input = input.dimshuffle(1, 0, 2)

        # When theano.scan calls step, input_n will be (n_batch, 4*num_units).
        # We define a slicing function that extract the input to each LSTM gate
        def slice_w(x, n):
            s = x[:, n * self.num_units:(n + 1) * self.num_units]
            if self.num_units == 1:
                s = T.addbroadcast(s, 1)  # Theano cannot infer this by itself
            return s

        # Create single recurrent computation step function
        # input_n is the n'th vector of the input
        def step(input_n, cell_previous, hid_previous, *args):

            # Compute the input-to-hidden activation
            if not self.precompute_input:
                input_n = T.dot(input_n, W_in_stacked)

            # Compute the hidden-to-hidden activation
            hid_to_hid = T.dot(hid_previous, W_hid_stacked)

            # Compute the second order term
            second_order_term = a_g_stacked * input_n * hid_to_hid

            # Compute the first order input-to-hidden term
            f_o_in_to_hid = b_g_in_to_hid_stacked * input_n

            # Compute the first order hidden-to-hidden term
            f_o_hid_to_hid = b_g_hid_to_hid_stacked * hid_to_hid

            # Calculate gates pre-activations and slice
            gates = (second_order_term + f_o_in_to_hid + f_o_hid_to_hid +
                     b_stacked)

            # Clip gradients
            if self.grad_clipping:
                gates = theano.gradient.grad_clip(
                    gates, -self.grad_clipping, self.grad_clipping)

            # Extract the pre-activation gate values
            ingate = slice_w(gates, 0)
            forgetgate = slice_w(gates, 1)
            cell_input = slice_w(gates, 2)
            outgate = slice_w(gates, 3)

            # Apply nonlinearities
            ingate = self.nonlinearity_ingate(ingate)
            forgetgate = self.nonlinearity_forgetgate(forgetgate)
            cell_input = self.nonlinearity_cell(cell_input)

            # Compute new cell value
            cell = forgetgate * cell_previous + ingate * cell_input

            outgate = self.nonlinearity_outgate(outgate)

            # Compute new hidden unit activation
            hid = outgate * self.nonlinearity(cell)
            return [cell, hid]

        def step_masked(input_n, mask_n, cell_previous, hid_previous, *args):
            cell, hid = step(input_n, cell_previous, hid_previous, *args)

            # Skip over any input with mask 0 by copying the previous
            # hidden state; proceed normally for any input with mask 1.
            cell = T.switch(mask_n, cell, cell_previous)
            hid = T.switch(mask_n, hid, hid_previous)

            return [cell, hid]

        if mask is not None:
            # mask is given as (batch_size, seq_len). Because scan iterates
            # over first dimension, we dimshuffle to (seq_len, batch_size) and
            # add a broadcastable dimension
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, mask]
            step_fun = step_masked
        else:
            sequences = input
            step_fun = step

        ones = T.ones((num_batch, 1))
        if not isinstance(self.cell_init, Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            cell_init = T.dot(ones, self.cell_init)

        if not isinstance(self.hid_init, Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            hid_init = T.dot(ones, self.hid_init)

        # The hidden-to-hidden weight matrix and the gating biases
        # are always used in this step and the bias
        non_seqs = [W_hid_stacked, a_g_stacked, b_g_in_to_hid_stacked,
                    b_g_hid_to_hid_stacked, b_stacked]

        # When we aren't precomputing the input outside of scan, we need to
        # provide the input weights and biases to the step function
        if not self.precompute_input:
            non_seqs += [W_in_stacked]

        if self.unroll_scan:
            # Retrieve the dimensionality of the incoming layer
            input_shape = self.input_shapes[0]
            # Explicitly unroll the recurrence instead of using scan
            cell_out, hid_out = unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[cell_init, hid_init],
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=input_shape[1])
        else:
            # Scan op iterates over first dimension of input and repeatedly
            # applies the step function
            cell_out, hid_out = theano.scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[cell_init, hid_init],
                go_backwards=self.backwards,
                truncate_gradient=self.gradient_steps,
                non_sequences=non_seqs,
                strict=True)[0]

        # When it is requested that we only return the final sequence step,
        # we need to slice it out immediately after scan is applied
        if self.only_return_final:
            hid_out = hid_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            hid_out = hid_out.dimshuffle(1, 0, 2)

            # if scan is backward reverse the output
            if self.backwards:
                hid_out = hid_out[:, ::-1]

        return hid_out


class MIGRULayer(MergeLayer):

    def __init__(self, incoming, num_units,
                 resetgate=MIGate(),
                 updategate=MIGate(),
                 hidden_update=MIGate(nonlinearity=nonlinearities.tanh),
                 hid_init=init.Constant(0.),
                 backwards=False,
                 learn_init=False,
                 gradient_steps=-1,
                 grad_clipping=0,
                 unroll_scan=False,
                 precompute_input=True,
                 mask_input=None,
                 only_return_final=False,
                 bn=False,
                 **kwargs):

        # This layer inherits from a MergeLayer, because it can have three
        # inputs - the layer input, the mask and the initial hidden state.  We
        # will just provide the layer input as incomings, unless a mask input
        # or initial hidden state was provided.
        incomings = [incoming]
        self.mask_incoming_index = -1
        self.hid_init_incoming_index = -1
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = len(incomings) - 1
        if isinstance(hid_init, Layer):
            incomings.append(hid_init)
            self.hid_init_incoming_index = len(incomings) - 1

        # Initialize parent layer
        super(MIGRULayer, self).__init__(incomings, **kwargs)

        self.learn_init = learn_init
        self.num_units = num_units
        self.grad_clipping = grad_clipping
        self.backwards = backwards
        self.gradient_steps = gradient_steps
        self.unroll_scan = unroll_scan
        self.precompute_input = precompute_input
        self.only_return_final = only_return_final

        if unroll_scan and gradient_steps != -1:
            raise ValueError(
                "Gradient steps must be -1 when unroll_scan is true.")

        # Retrieve the dimensionality of the incoming layer
        input_shape = self.input_shapes[0]

        if unroll_scan and input_shape[1] is None:
            raise ValueError("Input sequence length cannot be specified as "
                             "None when unroll_scan is True")

        # Input dimensionality is the output dimensionality of the input layer
        num_inputs = np.prod(input_shape[2:])

        def add_gate_params(gate, gate_name):
            """ Convenience function for adding layer parameters from a Gate
            instance. """

            W_in = self.add_param(gate.W_in, (num_inputs, num_units),
                                  name="W_in_to_{}".format(gate_name))
            W_hid = self.add_param(gate.W_hid, (num_units, num_units),
                                   name="W_hid_to_{}".format(gate_name))
            b = self.add_param(gate.b, (num_units,),
                               name="b_{}".format(gate_name),
                               regularizable=False)
            if gate.a_g is not None:
                a_g = self.add_param(gate.a_g, (num_units,),
                                     name="a_g_to_{}".format(gate_name),
                                     regularizable=False,
                                     trainable=gate.learn_a_g)
            else:
                a_g = T.zeros((num_units, ))

            if gate.b_g_in_to_hid is not None:
                b_g_in_to_hid = self.add_param(
                    gate.b_g_in_to_hid, (num_units,),
                    name="b_g_in_to_hid_to_{}".format(gate_name),
                    regularizable=False,
                    trainable=gate.learn_b_g_in_to_hid)
            else:
                b_g_in_to_hid = None

            if gate.b_g_hid_to_hid is not None:
                b_g_hid_to_hid = self.add_param(
                    gate.b_g_hid_to_hid, (num_units, ),
                    name="b_g_hid_to_hid_to_{}".format(gate_name),
                    regularizable=False,
                    trainable=gate.learn_b_g_hid_to_hid)
            else:
                b_g_hid_to_hid = None

            return (W_in, W_hid, b, a_g, b_g_in_to_hid, b_g_hid_to_hid,
                    gate.nonlinearity)

        # Add in all parameters from gates
        (self.W_in_to_updategate, self.W_hid_to_updategate, self.b_updategate,
         self.a_g_updategate, self.b_g_in_to_hid_updategate,
         self.b_g_hid_to_hid_updategate,
         self.nonlinearity_updategate) = add_gate_params(updategate,
                                                         'updategate')
        (self.W_in_to_resetgate, self.W_hid_to_resetgate, self.b_resetgate,
         self.a_g_resetgate, self.b_g_in_to_hid_resetgate,
         self.b_g_hid_to_hid_resetgate,
         self.nonlinearity_resetgate) = add_gate_params(resetgate,
                                                        'resetgate')
        (self.W_in_to_hidden_update, self.W_hid_to_hidden_update,
         self.b_hidden_update, self.a_g_hidden_update,
         self.b_g_in_to_hid_hidden_update, self.b_g_hid_to_hid_hidden_update,
         self.nonlinearity_hidden_update) = add_gate_params(hidden_update,
                                                            'hidden_update')

        # Initialize hidden state
        if isinstance(hid_init, Layer):
            self.hid_init = hid_init
        else:
            self.hid_init = self.add_param(
                hid_init, (1, self.num_units), name="hid_init",
                trainable=learn_init, regularizable=False)

        if bn:
            # create BN layer for correct input shape
            self.bn = BatchNormLayer(input_shape, axes=(0, 1))
            self.params.update(self.bn.params)  # make BN params your params
        else:
            self.bn = False

    def get_output_shape_for(self, input_shapes):
        # The shape of the input to this layer will be the first element
        # of input_shapes, whether or not a mask input is being used.
        input_shape = input_shapes[0]
        # When only_return_final is true, the second (sequence step) dimension
        # will be flattened
        if self.only_return_final:
            return input_shape[0], self.num_units
        # Otherwise, the shape will be (n_batch, n_steps, num_units)
        else:
            return input_shape[0], input_shape[1], self.num_units

    def get_output_for(self, inputs, **kwargs):
        """
        Compute this layer's output function given a symbolic input variable

        Parameters
        ----------
        inputs : list of theano.TensorType
            `inputs[0]` should always be the symbolic input variable.  When
            this layer has a mask input (i.e. was instantiated with
            `mask_input != None`, indicating that the lengths of sequences in
            each batch vary), `inputs` should have length 2, where `inputs[1]`
            is the `mask`.  The `mask` should be supplied as a Theano variable
            denoting whether each time step in each sequence in the batch is
            part of the sequence or not.  `mask` should be a matrix of shape
            ``(n_batch, n_time_steps)`` where ``mask[i, j] = 1`` when ``j <=
            (length of sequence i)`` and ``mask[i, j] = 0`` when ``j > (length
            of sequence i)``. When the hidden state of this layer is to be
            pre-filled (i.e. was set to a :class:`Layer` instance) `inputs`
            should have length at least 2, and `inputs[-1]` is the hidden state
            to prefill with.

        Returns
        -------
        layer_output : theano.TensorType
            Symbolic output variable.
        """
        # Retrieve the layer input
        input = inputs[0]
        # Retrieve the mask when it is supplied
        mask = None
        hid_init = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]
        if self.hid_init_incoming_index > 0:
            hid_init = inputs[self.hid_init_incoming_index]

        # Treat all dimensions after the second as flattened feature dimensions
        if input.ndim > 3:
            input = T.flatten(input, 3)

        if self.bn:
            input = self.bn.get_output_for(input)

        # Because scan iterates over the first dimension we dimshuffle to
        # (n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0, 2)
        seq_len, num_batch, _ = input.shape

        # Stack input weight matrices into a (num_inputs, 3*num_units)
        # matrix, which speeds up computation
        W_in_stacked = T.concatenate(
            [self.W_in_to_resetgate, self.W_in_to_updategate,
             self.W_in_to_hidden_update], axis=1)

        # Same for hidden weight matrices
        W_hid_stacked = T.concatenate(
            [self.W_hid_to_resetgate, self.W_hid_to_updategate,
             self.W_hid_to_hidden_update], axis=1)

        # Stack gate biases into a (3*num_units) vector
        b_stacked = T.concatenate(
            [self.b_resetgate, self.b_updategate,
             self.b_hidden_update], axis=0)

        # Stack second order gating biases into a (3*num_units) vector
        a_g_stacked = T.concatenate(
            [self.a_g_resetgate, self.a_g_updategate,
             self.a_g_hidden_update], axis=0)

        # Stack second order gating biases into a (3*num_units) vector
        b_g_in_to_hid_stacked = T.concatenate(
            [self.b_g_in_to_hid_resetgate, self.b_g_in_to_hid_updategate,
             self.b_g_in_to_hid_hidden_update], axis=0)

        # Stack second order gating biases into a (3*num_units) vector
        b_g_hid_to_hid_stacked = T.concatenate(
            [self.b_g_hid_to_hid_resetgate, self.b_g_hid_to_hid_updategate,
             self.b_g_hid_to_hid_hidden_update], axis=0)

        if self.precompute_input:
            # precompute_input inputs*W. W_in is (n_features, 3*num_units).
            # input is then (n_batch, n_time_steps, 3*num_units).
            input = T.dot(input, W_in_stacked)

        # When theano.scan calls step, input_n will be (n_batch, 3*num_units).
        # We define a slicing function that extract the input to each GRU gate
        def slice_w(x, n):
            s = x[:, n * self.num_units:(n + 1) * self.num_units]
            if self.num_units == 1:
                s = T.addbroadcast(s, 1)  # Theano cannot infer this by itself
            return s

        # Create single recurrent computation step function
        # input__n is the n'th vector of the input
        def step(input_n, hid_previous, *args):
            # Compute W_{hr} h_{t - 1}, W_{hu} h_{t - 1}, and W_{hc} h_{t - 1}
            hid_input = T.dot(hid_previous, W_hid_stacked)

            if self.grad_clipping:
                input_n = theano.gradient.grad_clip(
                    input_n, -self.grad_clipping, self.grad_clipping)
                hid_input = theano.gradient.grad_clip(
                    hid_input, -self.grad_clipping, self.grad_clipping)

            if not self.precompute_input:
                # Compute W_{xr}x_t + b_r, W_{xu}x_t + b_u, and W_{xc}x_t + b_c
                input_n = T.dot(input_n, W_in_stacked)

            # Compute the second_order_term
            second_order_term = a_g_stacked * input_n * hid_input

            # Compute the first order input-to-hidden term
            f_o_input = b_g_in_to_hid_stacked * input_n + b_stacked

            # Compute the first order hidden-to-hidden term
            f_o_hid_input = b_g_hid_to_hid_stacked * hid_input

            # Reset and update gates
            resetgate = (slice_w(second_order_term, 0) + slice_w(f_o_hid_input, 0) +
                         slice_w(f_o_input, 0))
            updategate = (slice_w(second_order_term, 1) + slice_w(f_o_hid_input, 1) +
                          slice_w(f_o_input, 1))
            resetgate = self.nonlinearity_resetgate(resetgate)
            updategate = self.nonlinearity_updategate(updategate)

            # Compute
            # (W_{xc}x_t + r_t \odot (W_{hc} h_{t - 1}) +
            #  r_t \odot (W_{xc}x_t * W_{hc} h_{t-1}))
            # This is different from the paper, but follows the
            # formulation used in Lasagne
            hidden_update_in = slice_w(f_o_hid_input, 2)
            hidden_update_hid = slice_w(f_o_hid_input, 2)
            hidden_update_s_o = slice_w(second_order_term, 2)
            hidden_update = (hidden_update_in +
                             resetgate * (hidden_update_hid + hidden_update_s_o))
            if self.grad_clipping:
                hidden_update = theano.gradient.grad_clip(
                    hidden_update, -self.grad_clipping, self.grad_clipping)
            hidden_update = self.nonlinearity_hidden_update(hidden_update)

            # Compute (1 - u_t)h_{t - 1} + u_t c_t
            hid = (1 - updategate) * hid_previous + updategate * hidden_update
            return hid

        def step_masked(input_n, mask_n, hid_previous, *args):
            hid = step(input_n, hid_previous, *args)

            # Skip over any input with mask 0 by copying the previous
            # hidden state; proceed normally for any input with mask 1.
            hid = T.switch(mask_n, hid, hid_previous)

            return hid

        if mask is not None:
            # mask is given as (batch_size, seq_len). Because scan iterates
            # over first dimension, we dimshuffle to (seq_len, batch_size) and
            # add a broadcastable dimension
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, mask]
            step_fun = step_masked
        else:
            sequences = [input]
            step_fun = step

        if not isinstance(self.hid_init, Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            hid_init = T.dot(T.ones((num_batch, 1)), self.hid_init)

        # The hidden-to-hidden weight matrix is always used in step
        non_seqs = [W_hid_stacked, a_g_stacked, b_g_in_to_hid_stacked,
                    b_g_hid_to_hid_stacked, b_stacked]
        # When we aren't precomputing the input outside of scan, we need to
        # provide the input weights and biases to the step function
        if not self.precompute_input:
            non_seqs += [W_in_stacked]

        if self.unroll_scan:
            # Retrieve the dimensionality of the incoming layer
            input_shape = self.input_shapes[0]
            # Explicitly unroll the recurrence instead of using scan
            hid_out = unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[hid_init],
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=input_shape[1])[0]
        else:
            # Scan op iterates over first dimension of input and repeatedly
            # applies the step function
            hid_out = theano.scan(
                fn=step_fun,
                sequences=sequences,
                go_backwards=self.backwards,
                outputs_info=[hid_init],
                non_sequences=non_seqs,
                truncate_gradient=self.gradient_steps,
                strict=True)[0]

        # When it is requested that we only return the final sequence step,
        # we need to slice it out immediately after scan is applied
        if self.only_return_final:
            hid_out = hid_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            hid_out = hid_out.dimshuffle(1, 0, 2)

            # if scan is backward reverse the output
            if self.backwards:
                hid_out = hid_out[:, ::-1]

        return hid_out
