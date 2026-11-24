#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Kiwon Kwon
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Build resnet8.tflite, the ResNet-8 of M3, trained on CIFAR-10.

The network has the structure of MLPerf Tiny's image classification model
(mlcommons/tiny, benchmark/training/image_classification/keras_model.py,
resnet_v1_eembc with its default of 26 filters; docs/design.md 6.6),
written anew here: a 32x32x3 input, conv3x3(26) with batch normalization
and ReLU, three residual stages of 26, 52 and 104 filters, 8x8 average
pooling, flatten, dense 10 and softmax. In a stage, conv3x3, batch
normalization, ReLU, conv3x3 and batch normalization are added to the
stage's input, through a 1x1 convolution of stride 2 in the second and
third stages (whose first conv3x3 also has stride 2), and ReLU follows the
sum. The convolutions pad to the same size and have He-normal initial
weights and an L2 penalty of 1e-4.

Training follows MLPerf Tiny's train.py: batch 32, Adam with a learning
rate of 0.001 times 0.99 per epoch, 500 epochs, random rotation (up to 15
degrees), translation (up to 0.1 of the image) and horizontal flips, and
pixels 0 to 255 as they are. train.py augments with Keras's legacy
ImageDataGenerator; Keras 3's random layers do the same here, filling with
the nearest pixel as ImageDataGenerator does by default. Training runs on
the CPU with a fixed seed and op determinism. The TFLite converter then
quantizes the model fully to int8 (int8 input and output), calibrated with
CIFAR-10 training images as models/mnist/make_model.py does (MLPerf Tiny
calibrates with chosen test images).

The script prints the accuracy of the float model and of the int8 model on
the 10,000 test images. --save-weights keeps the trained float weights and
--load-weights reads them instead of training, so that the conversion can
be redone alone.

Keras downloads CIFAR-10 and checks its SHA-256. CIFAR-10 is described in
Alex Krizhevsky, "Learning Multiple Layers of Features from Tiny Images",
2009, which its page asks users to cite; it states no license. Its images
are therefore not in this repository, and the model trained on it is under
this repository's license (docs/design.md 12.4).

The script is run by hand and its output is committed, so the compiler path
and CI need no TensorFlow (docs/design.md D12). It was run with TensorFlow
2.21.0 (Keras 3.15.1) on Python 3.12 (x86_64, 24 threads); another
TensorFlow version, machine or thread count may give other weights.

    python3 models/resnet8/make_model.py [-o models/resnet8]
"""

import argparse
import os

import numpy as np
import tensorflow as tf

SEED = 20261124
TF_VERSION = "2.21.0"

FILTERS = 26
EPOCHS = 500
BATCH_SIZE = 32
LEARNING_RATE = 0.001
LEARNING_RATE_DECAY = 0.99   # per epoch
L2_PENALTY = 1e-4
CALIBRATION_SAMPLES = 256


def conv(x, filters, size, stride, name):
    return tf.keras.layers.Conv2D(
        filters, size, strides=stride, padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=tf.keras.regularizers.l2(L2_PENALTY),
        name=name)(x)


def build_model(batch_size=None):
    """The network; batch_size=1 gives the static shapes TFLite Micro needs."""
    layers = tf.keras.layers
    inp = tf.keras.Input(shape=(32, 32, 3), batch_size=batch_size,
                         name="input")
    x = conv(inp, FILTERS, 3, 1, "conv")
    x = layers.ReLU(name="relu")(layers.BatchNormalization(name="bn")(x))
    for stage in range(3):
        filters = FILTERS << stage
        stride = 1 if stage == 0 else 2
        s = f"stage{stage + 1}_"
        y = conv(x, filters, 3, stride, s + "conv1")
        y = layers.BatchNormalization(name=s + "bn1")(y)
        y = layers.ReLU(name=s + "relu1")(y)
        y = conv(y, filters, 3, 1, s + "conv2")
        y = layers.BatchNormalization(name=s + "bn2")(y)
        if stage:
            x = conv(x, filters, 1, stride, s + "shortcut")
        x = layers.Add(name=s + "add")([x, y])
        x = layers.ReLU(name=s + "relu")(x)
    x = layers.AveragePooling2D(8, name="pool")(x)
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(10, kernel_initializer="he_normal", name="fc")(x)
    out = layers.Softmax(name="softmax")(x)
    return tf.keras.Model(inp, out, name="resnet8")


def load_cifar10():
    """CIFAR-10 as float32 NHWC images with pixels 0 to 255, and labels."""
    cifar10 = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    return (x_train.astype(np.float32), y_train.ravel(),
            x_test.astype(np.float32), y_test.ravel())


def train(x_train, y_train):
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()
    net = build_model()
    layers = tf.keras.layers
    augment = tf.keras.Sequential([
        layers.RandomRotation(15 / 360, fill_mode="nearest"),
        layers.RandomTranslation(0.1, 0.1, fill_mode="nearest"),
        layers.RandomFlip("horizontal"),
    ], name="augment")
    inp = tf.keras.Input(shape=(32, 32, 3))
    model = tf.keras.Model(inp, net(augment(inp)))
    model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    schedule = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: LEARNING_RATE * LEARNING_RATE_DECAY ** epoch)
    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
              callbacks=[schedule], verbose=2)
    return net


def convert(net, x_train):
    model = build_model(batch_size=1)
    model.set_weights(net.get_weights())
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
    ap.add_argument("--save-weights", metavar="FILE.weights.h5",
                    help="write the trained float weights")
    ap.add_argument("--load-weights", metavar="FILE.weights.h5",
                    help="read the float weights instead of training")
    args = ap.parse_args()

    tf.config.set_visible_devices([], "GPU")
    if tf.__version__ != TF_VERSION:
        print(f"warning: TensorFlow {tf.__version__}, the committed file "
              f"was made with {TF_VERSION}")
    x_train, y_train, x_test, y_test = load_cifar10()
    if args.load_weights:
        net = build_model()
        net.load_weights(args.load_weights)
    else:
        net = train(x_train, y_train)
    if args.save_weights:
        net.save_weights(args.save_weights)
    predictions = net.predict(x_test, batch_size=250, verbose=0)
    float_accuracy = float(np.mean(np.argmax(predictions, axis=1) == y_test))
    tflite_model = convert(net, x_train)
    int8_accuracy_, scale, zero_point = int8_accuracy(tflite_model, x_test,
                                                      y_test)

    path = os.path.join(args.output_dir, "resnet8.tflite")
    with open(path, "wb") as f:
        f.write(tflite_model)
    print(f"{path}: {len(tflite_model)} bytes, input scale {scale!r}, "
          f"zero point {zero_point}")
    print(f"test accuracy: float {float_accuracy:.4f}, "
          f"int8 {int8_accuracy_:.4f} ({len(y_test)} images)")


if __name__ == "__main__":
    main()
