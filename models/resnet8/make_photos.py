#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Kiwon Kwon
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Build the photograph test inputs of the ResNet-8 of M3 (photos/).

For each CIFAR-10 class the script downloads one photograph from Wikimedia
Commons (PHOTOS: its file name, author and SHA-256); --originals keeps the
downloads in a directory and uses them on later runs. Its author released
it under CC0-1.0, and Commons rates it a quality image. The photographs
were chosen before the model's classification of them was seen, with one
object of the class near the middle; a misclassified photograph stays.

Each photograph is decoded (JPEG with accurate integer IDCT), cropped to the
largest centered square and reduced to 32x32 by area averaging
(tf.image.resize, method "area"). The pixels are then quantized with the
input scale and zero point of resnet8.tflite, rounding once, and written as
photos/photo-<class>.bin: 3,072 int8 bytes, NHWC (RGB). The files stay under
CC0-1.0 (REUSE.toml, docs/design.md 12.4).

The script is run by hand and its output is committed, so the compiler path
and CI need no TensorFlow (docs/design.md D12). It was run with TensorFlow
2.21.0 on Python 3.12 (x86_64).

    python3 models/resnet8/make_photos.py [-o models/resnet8]
"""

import argparse
import hashlib
import os
import urllib.parse
import urllib.request

import numpy as np
import tensorflow as tf

TF_VERSION = "2.21.0"
USER_AGENT = "make_photos.py (CIFAR-10 class photographs for a model test)"
COMMONS = "https://commons.wikimedia.org/wiki/File:"
UPLOAD = "https://upload.wikimedia.org/wikipedia/commons/"

# (class, Commons file name, its directory under UPLOAD, author, SHA-256 of
# the original file)
PHOTOS = (
    ("airplane", "T-28 Trojan NAS Ft Worth Air Show 2016-3.jpg", "9/9d",
     "Balon Greyjoy",
     "809ccafa1c7aff62d587a64da2df912fb98e3ecd791fd777b9282cbe1213ab55"),
    ("automobile", "Avenue Roosevelt à Bruxelles, Zen Car en recharge.jpg",
     "9/91", "Benoît Prieur",
     "a96c87e05d5283a7e355ba87fed0ab9f27975c2a507b5453481c045d6471a6f0"),
    ("bird", "Jardim Zoológico de Curitiba 003.jpg", "e/e5", "Wilfredor",
     "d0e0832e30753148f1b46bb9bdff97bd61a4038e01ec149bdd9473d38efd1b6b"),
    ("cat", "Young tabby cat keeping watch.jpg", "9/95", "W.carter",
     "39512ca0b6dc81ed04a7900f8ea1bf3ce40004f276b75df301fb89ea60b240c6"),
    ("deer", "Reh (Mai 2025) 3.jpg", "3/3d", "Romzig",
     "432970a1dad2cbe7a98f8b449e3f94f7ee974cf483dc56ef604f85b1d37eaf96"),
    ("dog", "Dog at Nørre Vorupør Strand.jpg", "0/0d", "W.carter",
     "d08f9b67ffc35d60ea1cbf5b3e53ddb382ee4a16c5566e19d58fd6cdabe22c58"),
    ("frog", "Teichfrosch (Juni 2024).jpg", "0/05", "Romzig",
     "d4aeecd9ceac557c724f1c56253c46f7ebba9ee96c5450451e4fc0e990bab95a"),
    ("horse", "Horse in Tárcoles River.jpg", "3/37", "Bernard Gagnon",
     "3d574338f8cedab4fa91fc3ce2fe3ba54b2726a34e9d1ad46ffd83f3e3057024"),
    ("ship", "OSV Marsh Island.jpg", "3/31", "GuavaTrain",
     "295061b71a0ae7b59aed9056614748b4bbced6c1a08488cc2523431d8e1964c4"),
    ("truck", "UD Trucks Quon Dump CW 8L 6x4.jpg", "2/2d", "TTTNIS",
     "5e705f9e0ca6b3685320d1eee1b0a0f999f0035d3ee0f512c8159e7368475dc0"),
)


def original(name, directory, sha256, keep):
    """The original file, from the directory keep if it is there; else
    downloaded (and kept there when keep is given)."""
    quoted = urllib.parse.quote(name.replace(" ", "_"))
    page = COMMONS + quoted
    path = os.path.join(keep, name) if keep else None
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            data = f.read()
    else:
        req = urllib.request.Request(f"{UPLOAD}{directory}/{quoted}",
                                     headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
    if hashlib.sha256(data).hexdigest() != sha256:
        raise SystemExit(f"{page}: SHA-256 differs; this is not the file "
                         "chosen")
    if path and not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(data)
    return data, page


def reduce(data):
    """The largest centered square of a JPEG photograph as 32x32 float RGB
    pixels (0 to 255)."""
    image = tf.io.decode_jpeg(data, channels=3, dct_method="INTEGER_ACCURATE")
    h, w = image.shape[:2]
    s = min(h, w)
    top, left = (h - s) // 2, (w - s) // 2
    square = image[top:top + s, left:left + s]
    return tf.image.resize(square[tf.newaxis], (32, 32),
                           method="area")[0].numpy()


def quantize(x, scale, zero_point):
    q = np.round(x / scale) + zero_point
    return np.clip(q, -128, 127).astype(np.int8)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-o", "--output-dir", default=here)
    ap.add_argument("--model", default=os.path.join(here, "resnet8.tflite"),
                    help="the model whose input quantization is used")
    ap.add_argument("--originals", metavar="DIR",
                    help="keep the original files in DIR and use those "
                         "already there instead of downloading them")
    args = ap.parse_args()

    if tf.__version__ != TF_VERSION:
        print(f"warning: TensorFlow {tf.__version__}, the committed files "
              f"were made with {TF_VERSION}")
    interpreter = tf.lite.Interpreter(model_path=args.model)
    scale, zero_point = interpreter.get_input_details()[0]["quantization"]
    photos = os.path.join(args.output_dir, "photos")
    os.makedirs(photos, exist_ok=True)
    if args.originals:
        os.makedirs(args.originals, exist_ok=True)
    for cls, name, directory, author, sha256 in PHOTOS:
        data, page = original(name, directory, sha256, args.originals)
        path = os.path.join(photos, f"photo-{cls}.bin")
        with open(path, "wb") as f:
            f.write(quantize(reduce(data), scale, zero_point).tobytes())
        print(f"{path}: {page} ({author})")


if __name__ == "__main__":
    main()
