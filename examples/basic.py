import os
import numpy as np
import tflit


model_path = os.path.join(os.path.dirname(__file__), 'basic.tflite')
model = tflit.Model(model_path)
model.summary()

print('Input Shape:', model.input_shape)
print('Output Shape:', model.output_shape)
print('Dtype:', model.dtype)
print('Batch Size:', model.batch_size)

print('Input Dtypes:', model.input_dtypes)
print('Output Dtypes:', model.output_dtypes)

print('Input Names:', model.input_names)
print('Output Names:', model.output_names)

# test prediction
BATCH_SIZE = 16
X = np.random.randn(BATCH_SIZE, *model.input_shape[1:])
y_pred = model.predict(X)
print(X.shape, y_pred.shape)
assert y_pred.shape == (BATCH_SIZE,) + model.output_shape[1:]
