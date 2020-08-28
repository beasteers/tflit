import os
import glob
import json
import tflit
import pytest
import numpy as np


model_dir = os.path.join(os.path.dirname(__file__), 'models')
model_file = os.path.join(model_dir, '{}.tflite')
model_info_file = os.path.join(model_dir, '{}.json')



@pytest.mark.parametrize('name', [
    os.path.splitext(os.path.basename(f))[0]
    for f in glob.glob(model_file.format('*'))
])
def test_model(name):
    with open(model_info_file.format(name), 'r') as f:
        info = json.load(f)

    model = tflit.Model(model_file.format(name))
    model.summary()

    # check model shapes, replace batch dimension with 1
    assert replace_none(model.input_shape) == replace_none(info['input_shape'])
    assert replace_none(model.output_shape) == replace_none(info['output_shape'])

    # check names
    assert model.input_names == info['input_names']
    # assert model.output_names == info['output_names']

    assert model.dtype == info['dtype']

    # load arrays from json info
    to_array = lambda x: np.asarray(x, dtype=model.dtype)
    X_test = apply_maybe_list(to_array, info['X_test'], model.multi_input)
    y_pred = apply_maybe_list(to_array, info['y_pred'], model.multi_output)

    # predict on test data
    y_pred_tfl = model.predict_batch(X_test)

    # check outputs
    assert [y.shape for y in y_pred_tfl] == [y.shape for y in y_pred]
    assert np.allclose(
        np.asarray(list(flatten(y_pred_tfl))),
        np.asarray(list(flatten(y_pred))),
        rtol=1e-4, atol=1e-5)

    # apply batching
    faux_batch = lambda x, size=32: np.concatenate([x]*size)
    X_test_batch = apply_maybe_list(faux_batch, X_test, model.multi_input)
    y_pred_batch = apply_maybe_list(faux_batch, y_pred, model.multi_output)
    y_pred_batch_tfl = model.predict(X_test_batch)

    # check outputs
    assert [y.shape for y in y_pred_batch_tfl] == [y.shape for y in y_pred_batch]
    assert np.allclose(
        np.asarray(list(flatten(y_pred_batch_tfl))),
        np.asarray(list(flatten(y_pred_batch))),
        rtol=1e-4, atol=1e-5)

    return model




# Utilities


def replace_none(xs, value=1):
    return [
        replace_none(x, value) for x in xs
    ] if isinstance(xs, (list, tuple)) else (value if xs is None else xs)


def flatten(xs):
    try:
        yield from (xi for x in xs for xi in flatten(x))
    except TypeError:
        yield xs

def apply_maybe_list(func, lst, bool):
    return [func(x) for x in lst] if bool else func(lst)
