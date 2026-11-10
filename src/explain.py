"""Day 8: Grad-CAM, a heatmap that shows which parts of the picture the CNN used for its decision.

How it works, in short:
  1. Run the picture through the CNN and keep the output of the LAST convolution layer
     (a stack of small "feature maps" that mark where the filters found something).
  2. Ask: how much does the chosen class's score change if each feature map gets stronger?
     That is the gradient. Maps with a big positive gradient mattered for this decision.
  3. Weight the maps by that importance, add them up, keep only the positive part, and
     stretch the small result back to the picture size. Bright = important.

The catalog team can look at the heatmap to check the model is looking at the product,
not at a corner of the picture.
"""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import keras
import numpy as np
import tensorflow as tf


def last_conv_layer_name(model):
    """Find the name of the last convolution layer in the model."""
    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.Conv2D):
            return layer.name
    raise ValueError("This model has no convolution layer, so Grad-CAM cannot be used.")


_helper_models = {}      # remembers the helper model so we build it once, not once per picture


def build_helper(model):
    """A model that returns BOTH the last conv layer's output and the final prediction.

    A model loaded from disk has no connected output tensors in Keras 3, so we call its layers
    one by one on a fresh input. That reuses the trained weights and gives us the middle output."""
    if id(model) not in _helper_models:
        conv_name = last_conv_layer_name(model)
        new_input = keras.Input(shape=model.input_shape[1:])
        x = new_input
        for layer in model.layers:
            x = layer(x)
            if layer.name == conv_name:
                conv_output = x
        _helper_models[id(model)] = keras.Model(new_input, [conv_output, x])
    return _helper_models[id(model)]


def grad_cam(model, picture, class_index=None):
    """picture: grey (28, 28) array with values 0 to 1.

    Returns (heatmap, class_index). The heatmap has the same size as the picture, values 0 to 1.
    If class_index is None the model's own top guess is explained."""
    helper = build_helper(model)

    batch = tf.convert_to_tensor(picture[np.newaxis, ..., np.newaxis].astype("float32"))
    with tf.GradientTape() as tape:
        conv_output, prediction = helper(batch)
        if class_index is None:
            class_index = int(tf.argmax(prediction[0]))
        score = prediction[:, class_index]
    gradients = tape.gradient(score, conv_output)                    # how much each feature-map pixel matters

    importance = tf.reduce_mean(gradients, axis=(0, 1, 2))           # one weight per feature map
    heat = tf.reduce_sum(conv_output[0] * importance, axis=-1)       # weighted sum of the maps
    heat = tf.maximum(heat, 0).numpy()                               # keep only what pushes the score UP
    if heat.max() > 0:
        heat = heat / heat.max()                                     # scale to 0..1

    height, width = picture.shape
    return cv2.resize(heat, (width, height), interpolation=cv2.INTER_LINEAR), class_index


def heat_inside_mask(heatmap, mask):
    """Share of the heatmap's total brightness that falls on the product (0 to 1).
    Close to 1 means the model is looking at the product, not the background."""
    total = float(heatmap.sum())
    if total == 0:
        return 0.0
    return float(heatmap[mask > 0].sum()) / total


def overlay(picture_gray, heatmap):
    """Colour the heatmap and blend it over the picture. Returns a BGR picture (uint8)."""
    colour_heat = cv2.applyColorMap((heatmap * 255).astype("uint8"), cv2.COLORMAP_JET)
    base = cv2.cvtColor(picture_gray, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(base, 0.55, colour_heat, 0.45, 0)
