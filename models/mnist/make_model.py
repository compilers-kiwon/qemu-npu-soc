#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Kiwon Kwon
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Build mnist.tflite, the MNIST CNN of M2, and its digit test inputs.

The network is the one of docs/design.md 6.6: a 28x28x1 input,
conv3x3(8, VALID, ReLU), 2x2 average pooling, conv3x3(16, VALID, ReLU),
2x2 average pooling, flatten, dense 64 (ReLU), dense 10 and softmax. It is
trained on MNIST with a fixed seed and fully quantized to int8 by the TFLite
converter, calibrated with MNIST training images.

The script also writes one test image per digit (the first image of each
digit in the MNIST test set), quantized with the model's input scale and
zero point, as digits/digit-<d>.bin (784 int8 bytes, NHWC), and prints the
test-set accuracy of the float model and of the int8 model.

Keras downloads MNIST and checks its SHA-256. MNIST is Copyright Yann LeCun
and Corinna Cortes and licensed CC BY-SA 3.0; the digit images and the model
trained on it are declared under the same license (REUSE.toml,
docs/design.md 12.4).

The script is run by hand and its outputs are committed, so the compiler
path and CI need no TensorFlow (docs/design.md D12). It was run with
TensorFlow 2.21.0 on Python 3.12 (x86_64); another TensorFlow version may
produce different files.

    python3 models/mnist/make_model.py [-o models/mnist]
"""

import argparse
import os

import numpy as np
import tensorflow as tf

SEED = 20261123
TF_VERSION = "2.21.0"

EPOCHS = 2
BATCH_SIZE = 64
CALIBRATION_SAMPLES = 256


def build_model(batch_size=None):
    """The network; batch_size=1 gives the static shapes TFLite Micro needs."""
    inp = tf.keras.Input(shape=(28, 28, 1), batch_size=batch_size,
                         name="input")
    x = tf.keras.layers.Conv2D(8, 3, activation="relu", name="conv1")(inp)
    x = tf.keras.layers.AveragePooling2D(2, name="pool1")(x)
    x = tf.keras.layers.Conv2D(16, 3, activation="relu", name="conv2")(x)
    x = tf.keras.layers.AveragePooling2D(2, name="pool2")(x)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dense(64, activation="relu", name="fc1")(x)
    x = tf.keras.layers.Dense(10, name="fc2")(x)
    out = tf.keras.layers.Softmax(name="softmax")(x)
    return tf.keras.Model(inp, out, name="mnist")


def load_mnist():
    """MNIST as float32 NHWC images in [0, 1] and their labels."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    def images(x):
        return (x.astype(np.float32) / 255.0)[..., np.newaxis]

    return images(x_train), y_train, images(x_test), y_test


def train(x_train, y_train):
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()
    model = build_model()
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
              verbose=2)
    return model


def convert(trained, x_train):
    model = build_model(batch_size=1)
    model.set_weights(trained.get_weights())
    rng = np.random.default_rng(SEED)
    picks = rng.choice(len(x_train), CALIBRATION_SAMPLES, replace=False)

    def representative_dataset():
        for i in picks:
            yield [x_train[i:i + 1]]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def quantize(x, scale, zero_point):
    q = np.round(x / scale) + zero_point
    return np.clip(q, -128, 127).astype(np.int8)


def int8_accuracy(tflite_model, x_test, y_test):
    """Test-set accuracy of the int8 model; also its input quantization."""
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    scale, zero_point = inp["quantization"]
    correct = 0
    for x, y in zip(x_test, y_test):
        interpreter.set_tensor(inp["index"],
                               quantize(x[np.newaxis], scale, zero_point))
        interpreter.invoke()
        correct += int(np.argmax(interpreter.get_tensor(out["index"])[0]) == y)
    return correct / len(y_test), scale, zero_point


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-o", "--output-dir", default=here)
    args = ap.parse_args()

    if tf.__version__ != TF_VERSION:
        print(f"warning: TensorFlow {tf.__version__}, the committed files "
              f"were made with {TF_VERSION}")
    x_train, y_train, x_test, y_test = load_mnist()
    trained = train(x_train, y_train)
    _, float_accuracy = trained.evaluate(x_test, y_test, verbose=0)
    tflite_model = convert(trained, x_train)
    int8_accuracy_, scale, zero_point = int8_accuracy(tflite_model, x_test,
                                                      y_test)

    path = os.path.join(args.output_dir, "mnist.tflite")
    with open(path, "wb") as f:
        f.write(tflite_model)
    print(f"{path}: {len(tflite_model)} bytes, input scale {scale!r}, "
          f"zero point {zero_point}")
    digits = os.path.join(args.output_dir, "digits")
    os.makedirs(digits, exist_ok=True)
    for d in range(10):
        i = int(np.flatnonzero(y_test == d)[0])
        path = os.path.join(digits, f"digit-{d}.bin")
        with open(path, "wb") as f:
            f.write(quantize(x_test[i], scale, zero_point).tobytes())
        print(f"{path}: MNIST test image {i}")
    print(f"test accuracy: float {float_accuracy:.4f}, "
          f"int8 {int8_accuracy_:.4f} ({len(y_test)} images)")


if __name__ == "__main__":
    main()
