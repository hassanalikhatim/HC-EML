import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os

from utils.model_utils.model_architectures import *
from utils.model_utils.model import Keras_Model



def confirm_directory(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return

def study_model(model_under_study):
    base_weights = model_under_study.model.get_weights()
    for w,weights in enumerate(model_under_study.get_lite_weights):
        fig, axs = plt.subplots(1,4,figsize=(24,5))
        axs[0].hist(weights.reshape(-1), bins=50)
        axs[0].set_title(str(np.std(weights.reshape(-1))))
        axs[1].hist(base_weights[w].reshape(-1), bins=50)
        axs[1].set_title(str(np.std(base_weights[w].reshape(-1))))
        axs[2].scatter(base_weights[w].reshape(-1), weights.reshape(-1))
        axs[2].set_title(weights.dtype)
        axs[3].scatter(base_weights[w].reshape(-1), weights.reshape(-1)/base_weights[w].reshape(-1))
        axs[3].set_title(base_weights[w].dtype)
        

def debug_model(keras_model, debug_layer_number=2):
    model_debug = tf.keras.models.Model(inputs=keras_model.model.inputs, 
                                        outputs=keras_model.model.layers[debug_layer_number].output)
    keras_model_debug = Keras_Model(keras_model.path, data=keras_model.data, model=model_debug)
    keras_model_debug.model_name=keras_model.model_name+'_debug'
    keras_model_debug.prepare_and_save_tflite_model()
    keras_model_debug.compute_tflite_model_weights()

    tflite_predictions = keras_model_debug.tflite_predict(keras_model.data.x_test[:1])
    tf_predictions = model_debug.predict(keras_model.data.x_test[:1])

    shadow_model_debug = tf.keras.models.clone_model(keras_model.model)
    shadow_model_debug = tf.keras.models.Model(inputs=shadow_model_debug.inputs, 
                                              outputs=shadow_model_debug.layers[debug_layer_number].output)
    shadow_model_debug.set_weights(keras_model_debug.get_lite_weights)
    print("===========================")
    print("Differences: ", np.sum(np.abs(tflite_predictions-tf_predictions)), 
          np.sum(np.abs(tf_predictions-shadow_model_debug.predict(keras_model.data.x_test[:1]))))
    print("===========================")
    study_model(keras_model_debug)
    
    
def permute(X, y):
    perm = np.random.permutation(len(X))
    X = X[perm]
    y = y[perm]
    return X, y