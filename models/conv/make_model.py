#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Kiwon Kwon
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Build conv.tflite, the single-convolution test model of M1.7.

One Keras Conv2D (14x14x8 input, 16 filters, 3x3, SAME, stride 1, ReLU)
with seeded random weights and bias, fully quantized to int8 by the TFLite
converter with a seeded random representative dataset in [0, 1).

The script is run by hand and its output is committed, so the compiler
path and CI need no TensorFlow (docs/design.md D12). It was run with
TensorFlow 2.21.0 on Python 3.12 (x86_64); another TensorFlow version may
produce a different file.

    python3 models/conv/make_model.py [-o models/conv/conv.tflite]
"""

import argparse
import os

import numpy as np
import tensorflow as tf

SEED = 20261122
TF_VERSION = "2.21.0"

H, W, C = 14, 14, 8
OC = 16
KERNEL = 3
CALIBRATION_SAMPLES = 64


def build_model():
    tf.keras.utils.set_random_seed(SEED)
    inp = tf.keras.Input(shape=(H, W, C), batch_size=1, name="input")
    out = tf.keras.layers.Conv2D(
        OC,
        KERNEL,
        strides=1,
        padding="same",
        activation="relu",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED),
        bias_initializer=tf.keras.initializers.RandomUniform(
            -0.5, 0.5, seed=SEED + 1),
        name="conv",
    )(inp)
    return tf.keras.Model(inp, out, name="conv")


def representative_dataset():
    rng = np.random.default_rng(SEED)
    for _ in range(CALIBRATION_SAMPLES):
        yield [rng.random((1, H, W, C), dtype=np.float32)]


def convert(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-o", "--output",
                    default=os.path.join(here, "conv.tflite"))
    args = ap.parse_args()

    if tf.__version__ != TF_VERSION:
        print(f"warning: TensorFlow {tf.__version__}, the committed file "
              f"was made with {TF_VERSION}")
    data = convert(build_model())
    with open(args.output, "wb") as f:
        f.write(data)
    print(f"{args.output}: {len(data)} bytes")


if __name__ == "__main__":
    main()
