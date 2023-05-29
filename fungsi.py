import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout,
    LeakyReLU, Input, GlobalAveragePooling2D
)

def make_model():
    ModelDenseNet201 = tf.keras.models.Sequential([
        tf.keras.applications.DenseNet201(input_shape=(150, 150, 3),
                                          include_top=False,
                                          pooling='avg',
                                          weights='imagenet'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(32, activation=tf.keras.layers.ReLU()),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    return ModelDenseNet201

   