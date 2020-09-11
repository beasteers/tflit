import numpy as np
import tflite_runtime.interpreter as tflite
from . import util


class Model:

    def __init__(self, model_path, inputs=None, outputs=None, batch_size=None):
        self.model_path = model_path
        self.interpreter = tflite.Interpreter(model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # convert user inputted indexes to the actual tensor indexes
        self._input_idxs, self.multi_input = util.get_auto_index(
            inputs, self.input_details)
        self._output_idxs, self.multi_output = util.get_auto_index(
            outputs, self.output_details)

        # # update batch if provided
        # if batch_size:
        #     self.set_batch_size(batch_size)

    def __repr__(self):
        return '{}( {!r}, in={} out={} )'.format(
            self.__class__.__name__, self.model_path,
            self.input_shape, self.output_shape)

    # def set_batch_size(self, batch_size):
    #     # set batch for inputs
    #     for d in self.input_details:
    #         self.interpreter.resize_tensor_input(
    #             d['index'], [batch_size] + d['shape'][1:])
    #
    #     # set batch for outputs
    #     for d in self.output_details:
    #         self.interpreter.resize_tensor_input(
    #             d['index'], [batch_size] + d['shape'][1:])
    #
    #     # apply changes
    #     self.interpreter.allocate_tensors()
    #     self.batch_size = batch_size


    ##############
    # Model
    ##############

    def __call__(self, X, *a, **kw):
        return self.predict(X, *a, **kw)

    def predict_batch(self, X, multi_input=None, multi_output=None, add_batch=False):
        '''Predict a single batch.'''
        # set inputs
        X = self._check_inputs(X, multi_input)
        for i, idx in self._input_idxs:
            x = np.asarray(X[i], dtype=self.dtype)
            self.interpreter.set_tensor(idx, x[None] if add_batch else x)

        # compute outputs
        self.interpreter.invoke()

        # get outputs
        return self._check_outputs([
            self.interpreter.get_tensor(idx)
            for i, idx in self._output_idxs], multi_output)

    def predict(self, X, multi_input=None, multi_output=None):
        '''Predict data.'''
        return self._check_outputs([
            np.concatenate(x) for x in zip(*self.predict_each_batch(
                X, multi_input=multi_input, multi_output=True))
        ], multi=multi_output)

    def as_batches(self, X, multi_input=None, multi_output=None):
        '''Yield X in batches.'''
        X = self._check_inputs(X, multi_input)
        batch_size = self._check_batch_size(X)
        for i in range(0, len(X[0]), batch_size):
            xi = [x[i:i + batch_size] for x in X]
            yield xi if multi_output or len(xi) != 1 else xi[0]

    def predict_each_batch(self, X, multi_input=None, multi_output=None):
        '''Predict and yield each batch.'''
        # NOTE: multi=True so we don't squeeze
        for x in self.as_batches(X, multi_input=multi_input, multi_output=True):
            yield self.predict_batch(x, True, multi_output)


    def _check_inputs(self, X, multi=None):
        # coerce inputs to be a list
        return X if multi or self.multi_input else [X]

    def _check_outputs(self, Y, multi=None):
        # return either a single array or list of arrays depending on
        # single/multi output
        return (
            Y if multi or self.multi_output else
            Y[0] if len(Y) else None)

    def _check_batch_size(self, X):
        # check that there's only one batch size
        batch_sizes = [len(x) for x in X]
        if len(set(batch_sizes)) != 1:
            raise ValueError(
                'Expected a single batch size. Got {}.'.format(batch_sizes))
        return self.batch_size

    ##############
    # Info
    ##############

    # names

    @property
    def input_names(self):
        return [d['name'] for d in self.input_details]

    @property
    def output_names(self):
        return [d['name'] for d in self.output_details]

    # dtypes

    @property
    def input_dtypes(self):
        return [d['dtype'].__name__ for d in self.input_details]

    @property
    def output_dtypes(self):
        return [d['dtype'].__name__ for d in self.output_details]

    @property
    def dtype(self):
        dtypes = set(self.input_dtypes + self.output_dtypes)
        return next(iter(dtypes), None)

    # shapes

    @property
    def input_shapes(self):
        return [tuple(d['shape']) for d in self.input_details]

    @property
    def output_shapes(self):
        return [tuple(d['shape']) for d in self.output_details]

    # shape

    @property
    def input_shape(self):
        shape = self.input_shapes
        return shape[0] if len(shape) == 1 else shape

    @property
    def output_shape(self):
        shape = self.output_shapes
        return shape[0] if len(shape) == 1 else shape


    _batch_size = None
    @property
    def batch_size(self):
        if self._batch_size is None:
            self._batch_size = next((x[0] for x in self.input_shapes), None)
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value

    # print

    def summary(self):
        print(util.add_border('\n'.join([
            str(self),
            '', '-- Input details --',
            util.format_details(self.input_details),
            '', '-- Output details --',
            util.format_details(self.output_details),
        ]), ch='.'))
