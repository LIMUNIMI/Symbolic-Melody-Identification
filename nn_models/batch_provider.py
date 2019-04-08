#!/usr/bin/env python
"""
Classes defining methods to iterate batches for training neural networks
"""
import numpy as np


class RecurrentBatchProvider(object):
    """A class to load data from files and serve it in batches
       for sequential models
    """

    def __init__(self, dtype=np.float32):
        self.data = []
        self.sizes = []
        self.dtype = dtype

    def store_data(self, *args):

        if not all([len(x) == len(args[0]) for x in args]):
            raise Exception('The length of each array must be the same')

        self.n_inputs = len(args)

        dims = [None] * len(args)
        for arrays in zip(*args):

            for i, array in enumerate(arrays):
                if not np.all(dims[i] == array.shape[1:]):
                    if dims[i] is None:
                        dims[i] = array.shape[1:]
                    else:
                        raise Exception(
                            'Cannot deal with variable output shapes')

            self.data.append(arrays)
            self.sizes.append(len(arrays[0]))

        self.dims = dims
        self._cs = np.r_[0, np.cumsum(self.sizes)]

    def _make_batch_array(self, batch_size, segment_length, dim):
        return np.empty([batch_size, segment_length] + list(dim), dtype=self.dtype)

    def make_batch_arrays(self, batch_size, segment_length):
        return [self._make_batch_array(batch_size, segment_length, dim)
                for dim in self.dims]

    def iter_pieces(self):
        for arrays in self.data:
            yield (array[np.newaxis, :].astype(self.dtype, copy=False)
                   for array in arrays)

    def _get_batch(self, segment_producer, batch_size, segment_length,
                   batch_arrays=None):

        if batch_arrays is None:
            batch_arrays = self.make_batch_arrays(batch_size, segment_length)
        else:
            # Check that the number of given arrays is the same as the number of
            # inputs
            if len(batch_arrays) != self.n_inputs:
                raise Exception(('Different number of arrays provided: {0} given '
                                 'but {1} expected').format(len(batch_arrays),
                                                            self.n_inputs))

        for i, (piece, segment_end) in enumerate(segment_producer(batch_size,
                                                                  segment_length)):
            arrays = self.data[piece]

            start = segment_end - segment_length
            start_trimmed = max(0, start)

            for batch_a, array in zip(batch_arrays, arrays):
                batch_a[i, - (segment_end - start_trimmed):] = array[
                    start_trimmed: segment_end]

            if start < 0:
                for batch_a in batch_arrays:
                    batch_a[i, :- (segment_end - start_trimmed)] = 0

        return batch_arrays

    def _select_segments_start(self, k, segment_size):
        available_idx = np.array(self.sizes) - segment_size
        valid = np.where(available_idx >= 0)[0]
        try:
            piece_idx = valid[np.random.randint(0, len(valid), k)]
        except ValueError:
            raise Exception(("No sequence is in the dataset is long enough "
                             "to extract segments of length {}")
                            .format(segment_size))
        return np.column_stack(
            (piece_idx, np.ones(k, dtype=np.int) * segment_size))

    def _select_segments_end(self, k, segment_size):
        sizes = np.array(self.sizes)
        available_idx = sizes - segment_size
        valid = np.where(available_idx >= 0)[0]
        try:
            piece_idx = valid[np.random.randint(0, len(valid), k)]
        except ValueError:
            raise Exception(("No sequence is in the dataset is long enough "
                             "to extract segments of length {}")
                            .format(segment_size))

        return np.column_stack((piece_idx, sizes[piece_idx]))

    def _select_segments_valid(self, k, segment_size):
        available_idx = np.array(self.sizes) - segment_size + 1
        valid = np.where(available_idx > 0)[0]
        cum_idx = np.cumsum(available_idx[valid])

        try:
            segment_starts = np.random.randint(0, cum_idx[-1], k)
        except ValueError:
            raise Exception(("No sequence is in the dataset is long enough "
                             "to extract segments of length {}")
                            .format(segment_size))

        piece_idx = np.searchsorted(cum_idx - 1, segment_starts, side='left')
        index_within_piece = segment_starts - np.r_[0, cum_idx[:-1]][piece_idx]

        return np.column_stack(
            # (valid[piece_idx], index_within_piece))
            (valid[piece_idx], index_within_piece + segment_size))

    def _select_segments_full(self, k, segment_size):
        total_instances = self._cs[-1]
        segment_ends = np.random.randint(1, total_instances + 1, k)
        piece_idx = np.searchsorted(self._cs[1:], segment_ends, side='left')
        index_within_piece = segment_ends - self._cs[piece_idx]

        return np.column_stack((piece_idx, index_within_piece))

    def get_batch_full(self, batch_size, segment_length,
                       batch_arrays=None):
        """
        Return a batch from the stored data. The segments in the batch may start
        before the start of a data sequence. In this case, they are zero-padded
        on the left, up to the start of the sequence.

        Parameters
        ----------

        batch_size : int
            The number sequences to generate

        segment_length : int
            The desired length of the sequences

        batch_arrays : list of ndarrays, optional
            A list of arrays for storing the batch data in


        Returns
        -------

        tuple
            A tuple with  ndarrays, containing the data for the batch
        """
        return self._get_batch(self._select_segments_full, batch_size,
                               segment_length, batch_arrays)

    def get_batch_valid(self, batch_size, segment_length,
                        batch_arrays=None):
        """
        Return a batch from the stored data. Other than for `get_batch_full`, the
        segments in the batch are always the subseqeuence of a data sequence. No
        zero-padding will take place. Note that this implies that data from
        sequences shorter than `segment_length` will never appear in the
        returned batches.

        Parameters
        ----------

        batch_size : int
            The number sequences to generate

        segment_length : int
            The desired length of the sequences
        batch_arrays : list of ndarrays, optional
            A list of arrays for storing the batch data in


        Returns
        -------

        tuple
            A tuple with  ndarrays, containing the data for the batch
        """
        return self._get_batch(self._select_segments_valid, batch_size,
                               segment_length, batch_arrays)

    def get_batch_start(self, batch_size, segment_length,
                        batch_arrays=None):
        """
        Return a batch from the stored data. This function returns only segments
        starting at the beginning of a data sequence.

        Parameters
        ----------

        batch_size : int
            The number sequences to generate

        segment_length : int
            The desired length of the sequences

        batch_arrays : list of ndarrays, optional
            A list of arrays for storing the batch data in


        Returns
        -------

        tuple
            A tuple with  ndarrays, containing the data for the batch
        """
        return self._get_batch(self._select_segments_start, batch_size,
                               segment_length, batch_arrays)

    def get_batch_end(self, batch_size, segment_length,
                      batch_arrays=None):
        """
        Return a batch from the stored data. This function returns only segments
        ending at the end of a data sequence.

        Parameters
        ----------

        batch_size : int
            The number sequences to generate

        segment_length : int
            The desired length of the sequences

        batch_arrays : list of ndarrays, optional
            A list of arrays for storing the batch data in


        Returns
        -------

        tuple
            A tuple with  ndarrays, containing the data for the batch
        """
        return self._get_batch(self._select_segments_end, batch_size,
                               segment_length, batch_arrays)


if __name__ == '__main__':
    n_inputs = 2
    n_outputs = 2
    n_pieces = 3
    n_features = 4
    min_piece_len = 15
    max_piece_len = 30

    # create some data
    piece_lens = np.random.randint(min_piece_len,
                                   max_piece_len + 1,
                                   n_pieces)
    X = [np.column_stack((np.ones(n_instances) * i,
                          np.arange(n_instances))).astype(np.int)
         for i, n_instances in enumerate(piece_lens)]
    Y = [np.random.random((n_instances, n_outputs, n_features))
         for n_instances in piece_lens]
    # Y = X
    bp = RecurrentBatchProvider()
    bp.store_data(X, Y)

    for x, y in bp.iter_pieces():
        print x, y
