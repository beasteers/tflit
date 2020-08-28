# tflit 🔥
Because WTH `tflite_runtime`?

`interpreter.invoke()`?
`interpreter.set_tensor(input_details[0]['index'], X)`?

Having to select a platform-specific url from [here](https://www.tensorflow.org/lite/guide/python)?

Uh no. certainly not 🔥.

What this does:
 - Detects your platform + Python version so you don't have to pick the right url and you can add `tflite_runtime` as a dependency **without having to pick a single platform to support.**
 - Creates a familiar `keras`-like interface for models, so you can do `tflit.Model(path).predict(X)` without ever having to think about tensor indexes or three step predictions, or batching.


## Install

```bash
pip install tflit
```

## Usage

I tried to provide an interface as similar to Keras as possible.

```python
import tflit

model = tflit.Model('path/to/model.tflite')
model.summary()  # prints input and output details

print(model.input_shape)   # (10, 30)  - a single input
print(model.output_shape)  # [(5, 2), (1, 2)]  - two outputs
print(model.dtype)         # 'float32'

# *see notes below
print(model.input_names)   # may not preserve names (based on how you export)
print(model.output_names)  # doesn't preserve names atm

# predict over batches of outputs.
y_pred = model.predict(np.random.randn(32, 10, 30))

# predict single output at a time
y_pred = model.predict_batch(np.random.randn(1, 10, 30))
```

## Dark Ages

Just for reference, this is how I used to do it:

```python
def load_tflite_model_function(model_path, **kw):
    import tflite_runtime.interpreter as tflite
    compute = prepare_model_function(tflite.Interpreter(model_path), **kw)
    compute.model_path = model_path
    return compute


def prepare_model_function(model, verbose=False):
    # assumes a single input and output
    in_dets = model.get_input_details()[0]
    out_dets = model.get_output_details()[0]

    model.allocate_tensors()
    def compute(x):
        # set inputs
        model.set_tensor(in_dets['index'], X.astype(in_dets['dtype']))
        # compute outputs
        model.invoke()
        # get outputs
        return model.get_tensor(out_dets['index'])

    if verbose:
        print('-- Input details --')
        print(in_dets, '\n')
        print('-- Output details --')
        print(out_dets, '\n')

    # set input and output shapes so they're easily accessible
    compute.input_shape = in_dets['shape'][1:]
    compute.output_shape = out_dets['shape'][1:]
    return compute
```
This was cleaner than the code that I factored it out from, but it is still unnecessarily complex and I got tired after copying it over to my 3rd project. This also doesn't handle things like multiple inputs/outputs or batching.

## Notes
 - I was having trouble getting tflite_runtime to install as a dependency in `setup.py` so right now, it's just installing on first run if it's not already installed. I'll probably fix that at some point... but I have other things that I need to be doing and this is working atm. Hopefully tensorflow will just start deploying to pypi and this will all be resolved. Not sure what's going on there...

 - It's possible that `tflite_runtime` may not have a build for your system. Check [this](https://www.tensorflow.org/lite/guide/python) link to verify.

 - There's a bug with the current `tflite` converter where it doesn't copy over the input and output names.

    However, if you do this when you're exporting, the input names will be saved:
    ```python
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.experimental_new_converter = True  # <<< this
    tflite_model = converter.convert()
    ```

    But still no luck with the output names :/. To be clear, this is a tensorflow issue and I have no control over this.


 - I intended to have a `model.set_batch_size` method to change the batch size at runtime, but it doesn't currently work because tflite freaks out about there being an increased tensor size (it doesn't know how to broadcast). This is also a tensorflow issue.

    For the time being, we just compute one batch at a time and concatenate them at the end. If the model's fixed batch size doesn't divide evenly, it will throw an error. By default, tflite converts `None` batch sizes to `1` so most of the time it won't be a problem. To compute a single frame, it is more efficient to use `model.predict_batch(X)` directly.

I would love to get both of these resolved, but they are out of my control and I don't really have the bandwidth or the urgent need to have these resolved.
