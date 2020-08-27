import numpy as np
import tflite_runtime.interpreter as tflite


class Model:

    batch_size = 1

    def __init__(self, model_path, inputs=None, outputs=None, batch_size=None):
        self.model_path = model_path
        self.interpreter = tflite.Interpreter(model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # convert user inputted indexes to the actual tensor indexes
        self._input_idxs, self.multi_input = _get_auto_index(
            inputs, self.input_details)
        self._output_idxs, self.multi_output = _get_auto_index(
            outputs, self.output_details)

        # update batch if provided
        if batch_size:
            self.set_batch_size(batch_size)

    def __repr__(self):
        return '{}({}, in={} out={})'.format(
            self.__class__.__name__, self.model_path,
            self.input_shape, self.output_shape)

    def set_batch_size(self, batch_size):
        # set batch for inputs
        for d in self.input_details:
            self.interpreter.resize_tensor_input(
                d['index'], [batch_size] + d['shape'][1:])

        # set batch for outputs
        for d in self.output_details:
            self.interpreter.resize_tensor_input(
                d['index'], [batch_size] + d['shape'][1:])

        # apply changes
        self.interpreter.allocate_tensors()
        self.batch_size = batch_size


    ##############
    # Model
    ##############

    def predict(self, X, multi_input=None, multi_output=None, add_batch=False):
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

    def predict_batch(self, X, batch_size=None):
        X = self._check_inputs(X)
        batch_size = self._check_batch_size(X, batch_size)

        outs = [
            np.concatenate(x) for x in zip(*(
                self.predict([x[i][None] for x in X], True, True)
                for i in range(len(X[0]))
            ))
        ]
        return self._check_outputs(outs)

    def _check_inputs(self, X, multi=None):
        return X if multi or self.multi_input else [X]

    def _check_outputs(self, Y, multi=None):
        return (
            Y if multi or self.multi_output else
            Y[0] if Y else None)

    def _check_batch_size(self, X, batch_size=None):
        batch_sizes = [len(x) for x in X]
        # check that there's only one size
        unique_sizes = set(batch_sizes)
        if len(unique_sizes) != 1:
            raise ValueError(
                'Expected a single batch size. Got {}.'.format(batch_sizes))
        # check that it's the size we expect
        size = next(iter(unique_sizes))
        if batch_size and batch_size != size:
            raise ValueError('Expected a batch size of {}, got {}.'.format(
                batch_size, size))

        return size

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

    # print

    def summary(self):
        print('-' * 30)
        print('|', self)
        print('-- Input details --')
        print(format_details(self.input_details))
        print('-- Output details --')
        print(format_details(self.output_details))
        print('-' * 30)


_HIDDEN_DETAILS = ('name',)

def format_details(details, ignore=_HIDDEN_DETAILS):
    return '\n'.join(
        '  {}'.format(d['name']) + ''.join(
            '\n    {}: {}'.format(k, v)
            for k, v in d.items()
            if k not in ignore
        ) for d in details
    )


def _get_auto_index(idxs, details, multi_input=None):
    if isinstance(idxs, int):
        idxs = [idxs]
    elif idxs is None:
        idxs = list(range(len(details)))
    multi_input = len(idxs) != 1 if multi_input is None else multi_input

    idxs = [(i, details[i]['index']) for i in idxs]
    return (idxs if multi_input else idxs[:1]), multi_input
