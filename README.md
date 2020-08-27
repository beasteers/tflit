# tflit 🔥
Because WTH `tflite_runtime`?

`interpreter.invoke()`?
`interpreter.set_tensor(input_details[0]['index'], X)`?

Uh no. certainly not 🔥.


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

print(model.input_shape)
print(model.output_shape)
print(model.dtype)

# *see notes below
print(model.input_names)  # may not preserve
print(model.output_names)  # doesn't preserve

# predict single output at a time
y_pred = model.predict(np.random.randn(1, 10, 30))

# predict over a batch of outputs.
y_pred = model.predict_batch(np.random.randn(32, 10, 30))

```


## Notes
 - There's a bug with the current `tflite` converter where it doesn't copy over the input and output names.

    However, if you do this when you're exporting, the input names will be saved:
    ```python
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.experimental_new_converter = True  # <<< this
    tflite_model = converter.convert()
    ```

    But still no luck with the output names :/. To be clear, this is a tensorflow issue and I have no control over this.


 - I intended to have a `model.set_batch_size` method to change the batch size at runtime, but it doesn't currently work because tflite freaks out about there being an increased tensor size (it doesn't know how to broadcast). This is also a tensorflow issue.

    For the time being, I've added a `predict_batch` method which will compute for each item in the first dimension of each output.

I would love to get both of these resolved, but they are out of my control and I don't really have the bandwidth or the urgent need to have these resolved.
