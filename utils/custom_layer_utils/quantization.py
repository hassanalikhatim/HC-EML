from tensorflow import keras
import tensorflow as tf
import numpy as np
    


class Quantization(keras.layers.Layer):
    def __init__(self, quantization_levels=16):
        self.quantization_levels = quantization_levels
        super(Quantization, self).__init__()
        
    def call(self, inputs):
        inputs = tf.cast(inputs*self.quantization_levels, tf.dtypes.int32)
        return tf.cast(inputs, tf.dtypes.float32)/self.quantization_levels
    
    
class Random_Quantization(keras.layers.Layer):
    def __init__(self):
        super(Random_Quantization, self).__init__()
        
    def call(self, inputs):
        quantization_levels = tf.cast(tf.random.uniform([], 2, 16, tf.dtypes.int32), tf.dtypes.float32)
        inputs = tf.cast(inputs*quantization_levels, tf.int32)
        return tf.cast(inputs, tf.float32)/quantization_levels