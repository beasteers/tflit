import os
import json
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K
import tflit


def basic():
    inp = x = L.Input((10,), name='adsf')
    out = L.Dense(2, name='sdfg')(x)
    return tf.keras.Model(inp, out)

def basic2():
    inp = x = L.Input((100,))
    for i in range(5):
        x = L.Dense(50 - i * 5)(x)
    out = L.Dense(2)(x)
    return tf.keras.Model(inp, out)

def multi_input():
    inp1 = x = L.Input((10,), name='adsf')
    inp2 = x = L.Input((10,), name='adsf2')
    out = L.Dense(5, name='ddddd')(L.Add()([inp1, inp2]))
    return tf.keras.Model([inp1, inp2], out)

def multi_output():
    inp = x = L.Input((10,), name='adsf')
    out1 = L.Dense(2, name='sdfg')(x)
    out2 = L.Dense(5, name='sdfg2')(x)
    return tf.keras.Model(inp, [out1, out2])

def multi_in_out():
    inp1 = x = L.Input((10,), name='adsf')
    inp2 = x = L.Input((10,), name='adsf2')
    x = L.Add()([inp1, inp2])
    out1 = L.Dense(5, name='sdfg')(x)
    out2 = L.Dense(6, name='sdfg2')(x)
    return tf.keras.Model([inp1, inp2], [out1, out2])

# def multi_dtype():
#     inp1 = x = L.Input((10,), name='adsf', dtype=np.float32)
#     inp2 = x = L.Input((10,), name='adsf2', dtype=np.float64)  #K.cast(, np.float32)
#     x = L.Add()([inp1, inp2])
#     out1 = L.Dense(5, name='sdfg')(x)
#     out2 = L.Dense(6, name='sdfg2')(x)
#     # out2 = K.cast(out2, name='sdfg2')
#     return tf.keras.Model([inp1, inp2], [out1, out2])


MODEL_FUNCS = [
    basic, basic2, multi_input, multi_output, multi_in_out, 
    # multi_dtype
]


def replace_none(xs, value=1):
    return [
        replace_none(x, value) for x in xs
    ] if isinstance(xs, (list, tuple)) else (value if xs is None else xs)

def get_model_info(model):
    single_in = len(model.input_names) == 1
    single_out = len(model.output_names) == 1
    X_test = [
        np.random.randn(*(1 if x is None else x for x in sh)) for sh in
        ([model.input_shape] if single_in else model.input_shape)]
    y_pred = model.predict(X_test)
    return {
        'input_shape': model.input_shape,
        'output_shape': model.output_shape,
        'input_names': model.input_names,
        'output_names': model.output_names,
        'dtype': model.dtype,
        'X_test': (
            X_test[0].tolist() if single_in else
            [x.tolist() for x in X_test]),
        'y_pred': (
            y_pred.tolist() if single_out else
            [x.tolist() for x in y_pred]),
    }

model_dir = os.path.join(os.path.dirname(__file__), 'models')
model_file = os.path.join(model_dir, '{}.tflite')
model_info_file = os.path.join(model_dir, '{}.json')

def main():
    os.makedirs(model_dir, exist_ok=True)
    for func in MODEL_FUNCS:
        name = func.__name__
        # create the model
        model = func()
        model.summary()
        info = get_model_info(model)

        # Convert model
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.experimental_new_converter = True
        tflite_model = converter.convert()
        # litmodel = tflit.Model(model_content=tflite_model)
        print(type(tflite_model))
        # print(litmodel)

        # Save model.
        fname = model_file.format(name)
        with tf.io.gfile.GFile(fname, 'wb') as f:
            f.write(tflite_model)

        with open(model_info_file.format(name), 'w') as f:
            json.dump(info, f)

        print(f'Saved to {fname}.')


if __name__ == '__main__':
    import fire
    fire.Fire(main)
