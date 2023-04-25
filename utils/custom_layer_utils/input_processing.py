from tensorflow import keras
import tensorflow as tf
import numpy as np
    

class RGB2Gray3(keras.layers.Layer):
    def __init__(self):
        super(RGB2Gray3, self).__init__()

    def call(self, inputs):
        inputs = tf.reduce_mean(inputs, axis=3)
        inputs_2 = tf.stack([inputs, inputs, inputs], axis=3)
        return inputs_2
    

class Gray2Gray3(keras.layers.Layer):
    def __init__(self):
        super(Gray2Gray3, self).__init__()

    def call(self, inputs):
        inputs_2 = tf.stack([inputs, inputs, inputs], axis=3)
        return inputs_2
    
    
class RGB2Gray(keras.layers.Layer):
    def __init__(self):
        super(RGB2Gray, self).__init__()

    def call(self, inputs):
        inputs = tf.reduce_mean(inputs, axis=3)
        inputs = tf.expand_dims(inputs, axis=3)
        return inputs
    
    