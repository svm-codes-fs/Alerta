"""Step 1 of the Raspberry Pi plan: convert the eye-state CNN to TensorFlow Lite.

Run on the laptop (needs full TensorFlow):  python raspberry_pi/convert_to_tflite.py
Output: raspberry_pi/eye_model.tflite  (float16 weights, about half the size of the .h5)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import tensorflow as tf  # noqa: E402

from alerta.model_loader import load_eye_model  # noqa: E402

model = load_eye_model()
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
out = os.path.join(HERE, 'eye_model.tflite')
with open(out, 'wb') as f:
    f.write(converter.convert())
print(f'Saved {out} ({os.path.getsize(out) / 1e6:.1f} MB)')
