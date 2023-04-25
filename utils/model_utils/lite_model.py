import tensorflow as tf
import os
import numpy as np

from utils.model_utils.model_architectures import ae_by_hassan, ae_by_hassan_gray
from utils.attack_utils.attack import Attack
from utils.custom_layer_utils.quantization import Quantization
from utils.general_utils import confirm_directory



class Lite_Model:
    def __init__(self, keras_model, lite_name="dynamic", verbose=True):
        self.keras_model = keras_model
        self.lite_name = lite_name
        self.verbose = verbose
        
    
    def print_out(self, *print_statement, end=""):
        if self.verbose:
            print(*print_statement, end=end)
            
            
    def prepare_logits_model(self):
        self.logits_model = tf.keras.models.Model(inputs=self.model.input, 
                                                  outputs=self.model.get_layer("logits_layer").output)
    
        
    def prepare_and_save_tflite_interpreter(self):
        tflite_converter = tf.lite.TFLiteConverter.from_keras_model(self.keras_model.model)
        
        # Post-training float16 quantization
        if self.lite_name=='dynamic':
            print("Dynamic Quantization")
            tflite_converter.optimizations = [tf.lite.Optimize.DEFAULT]
        elif self.lite_name=='float16':
            print("Float16 Quantization")
            tflite_converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_converter.target_spec.supported_types = [tf.float16]
        elif self.lite_name=='integer':
            print("Integer Quantization")
            # Post-training integer quantization
            def representative_data_gen():
                for input_value in tf.data.Dataset.from_tensor_slices(self.keras_model.data.x_test).batch(1).take(100):
                    yield [input_value]
            tflite_converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_converter.representative_dataset = representative_data_gen
            tflite_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            # Set the input and output tensors to uint8 (APIs added in r2.3)
            tflite_converter.inference_input_type = tf.uint8
            tflite_converter.inference_output_type = tf.uint8
        
        lite_model = tflite_converter.convert()
        save_name = self.keras_model.model_name+'_'+self.lite_name+'.tflite'
        open(self.keras_model.path+'models/'+save_name, "wb").write(lite_model)
        self.interpreter = tf.lite.Interpreter(model_path=self.keras_model.path+'models/'+save_name)
    
    
    def get_tflite_model_size(self):
        model_size = os.path.getsize(self.keras_model.model_name+'.tflite')
        print('TF-Lite Model size: ' + str(round(model_size / 1024, 3)) + ' KB')
        
    
    def load_tflite_interpreter(self):
        save_name = self.keras_model.model_name+'_'+self.lite_name+'.tflite'
        tflite_path = self.keras_model.path+'models/'+save_name
        if os.path.isdir(tflite_path):
            self.interpreter = tf.lite.Interpreter(model_path=tflite_path)
        else:
            self.prepare_and_save_tflite_interpreter()
        
    
    def compute_tflite_model_weights(self):
        # Use `tf.lite.Interpreter` to load the written .tflite back from the file system.
        all_tensor_details = self.interpreter.get_tensor_details()
        self.interpreter.allocate_tensors()
        
        supported_layer_names = ['conv2d', 'dense', 'logits_layer']
        supported_tf_variable_names = {
            'conv2d': ['kernel', 'bias'],
            'dense': ['kernel', 'bias'],
            'logits_layer': ['kernel', 'bias']
        }
        supported_tflite_variable_names = {
            'conv2d': ['Conv2D', 'bias'],
            'dense': ['MatMul', 'bias'],
            'logits_layer': ['MatMul', 'bias']
        }
        
        weights_length = 0
        allowed_layers = []
        for layer in self.keras_model.model.layers:
          length = len(layer.trainable_variables)
          if length > 0:
            weights_length = weights_length + length
            allowed_layers.append(layer.name)
        self.print_out(allowed_layers)
        
        tflite_allowed_layer_names = []
        tflite_layer_names = []
        for tensor_detail in all_tensor_details:
          tflite_layer_names.append(tensor_detail['name'])
          for allowed_layer in allowed_layers:
            if allowed_layer in tensor_detail['name'].split('/'):
              if len(tflite_allowed_layer_names)<len(self.keras_model.model.get_weights()):
                tflite_allowed_layer_names.append(tensor_detail['name'])
        self.print_out(tflite_allowed_layer_names)
        self.print_out(tflite_layer_names)
        
        get_tflite_weights = []
        for layer in self.keras_model.model.layers:
            for supported_name in supported_layer_names:
                if supported_name in layer.name:
                    for variable,variable_name in enumerate(supported_tflite_variable_names[supported_name]):
                        weight_name = layer.name+'/'+variable_name
                        self.print_out("TF name: " + layer.name + '/' +
                                       supported_tf_variable_names[supported_name][variable], end="")
                        for tflite_layer_name in tflite_allowed_layer_names:
                            if weight_name in tflite_layer_name:
                                self.print_out("\t\t TFLITE name:", tflite_layer_name)
                                unscaled_weights = self.interpreter.tensor(
                                    tflite_layer_names.index(tflite_layer_name)
                                    )().astype('float32')
                                scales = all_tensor_details[
                                    tflite_layer_names.index(tflite_layer_name)
                                    ]['quantization_parameters']['scales']
                                if len(scales)>0:
                                    scales_shape = list(scales.shape)
                                    for i in unscaled_weights.shape[len(scales.shape):]:
                                        scales_shape.append(1)
                                    scales_shape = tuple(scales_shape)
                                    scales = scales.reshape(scales_shape)
                                    get_tflite_weights.append(unscaled_weights*scales)
                                else:
                                    get_tflite_weights.append(unscaled_weights)
        
        self.get_lite_weights = []
        for w,weight in enumerate(self.keras_model.model.get_weights()):
          if len(weight.shape)>1:
            # transposing_axes = list(np.arange(1,len(weight.shape)))+[0]
            # supported_weight = np.transpose(get_tflite_weights[w], transposing_axes)
            supported_weight = np.rollaxis(get_tflite_weights[w], 0, len(weight.shape)).astype('float32')
            self.get_lite_weights.append(supported_weight)
            self.print_out(weight.shape, supported_weight.shape, end="\t\t")
            self.print_out(np.sum(np.abs(supported_weight - weight)))
          else:
            self.get_lite_weights.append(get_tflite_weights[w])
            self.print_out(weight.shape, get_tflite_weights[w].shape, end="\t\t\t\t")
            self.print_out(np.sum(np.abs(weight-get_tflite_weights[w])))
    
    
    def prepare_tflite_model(self, lite_name):
        self.lite_name = lite_name
        self.load_tflite_interpreter()
        self.compute_tflite_model_weights()
        self.model = self.keras_model.model_architecture(self.keras_model.data)
        self.model.set_weights(self.get_lite_weights)
        self.prepare_logits_model()
        
        
    def tflite_predict(self, inputs):
        self.load_tflite_model()
        all_tensor_details = self.interpreter.get_tensor_details()
        
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        self.interpreter.resize_tensor_input(input_details[0]['index'], (len(inputs), 128, 128, 3))
        self.interpreter.resize_tensor_input(output_details[0]['index'], (len(inputs), 7))
        self.interpreter.allocate_tensors()
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        self.interpreter.set_tensor(input_details[0]['index'], inputs)
        self.interpreter.invoke()
        
        tflite_model_predictions = self.interpreter.get_tensor(output_details[0]['index'])
        return tflite_model_predictions
        
        
    def compute_tflite_model_accuracy(self):
        from sklearn.metrics import accuracy_score
        
        tflite_predictions = self.tflite_predict(self.keras_model.data.x_test)
        prediction_classes = np.argmax(tflite_predictions, axis=1)
        acc = accuracy_score(prediction_classes, np.argmax(self.keras_model.data.y_test, axis=1))
        print("The accuracy of the model is: ", acc)
        
        
    def compute_adversarial_images(self, N=100, attack_name='pgd',
                                      epsilon=0.01):
        attack = Attack(self.keras_model.data, self)
        adversarial_images = attack.evaluate_on_attack(N, epsilon, attack_name=attack_name)
        self.model.evaluate(adversarial_images, self.keras_model.data.y_test[:N])
        return adversarial_images
    
        
    def evaluate_quantized_model_accuracy(self, adversarial_images,
                                          quantization_levels=([2, 4, 8, 16, 255])):
        clean_accuracies = []
        adv_accuracies = []
        for quantization_level in quantization_levels:
            quantized_model = tf.keras.Sequential()
            quantized_model.add(tf.keras.layers.Input(shape=self.keras_model.data.get_input_shape()))
            quantized_model.add(Quantization(quantization_levels=quantization_level))
            quantized_model.add(self.model)
            quantized_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            clean_acc = quantized_model.evaluate(self.keras_model.data.x_test, 
                                                 self.keras_model.data.y_test, 
                                                 verbose=False)[1]
            adv_acc = quantized_model.evaluate(adversarial_images, 
                                               self.keras_model.data.y_test[:len(adversarial_images)], 
                                               verbose=False)[1]
            print("Evaluating for quantization level: ", quantization_level)
            self.print_out("Accuracy on clean examples: ", clean_acc)
            self.print_out("Accuracy on perturbed adversarial examples: ", adv_acc)
            clean_accuracies.append(clean_acc)
            adv_accuracies.append(adv_acc)
        print("Summary")
        print("Clean accuraces: ", clean_accuracies)
        print("Adversarial accuracies: ", adv_accuracies)
                  
    
    def perform_adversarial_analysis(self, attack_name='pgd', epsilon=0.01):
        save_dir = self.keras_model.model_name+'/'
        save_name = save_dir + 'adversarial_images_' + attack_name
        save_name = save_name + '('+str(epsilon)+')_' + self.lite_name
        try:
            adversarial_images = np.load(self.keras_model.path+'adversarial_images/'+save_name+'.npy')
            print(adversarial_images.shape)
        except:
            adversarial_images = self.compute_adversarial_images(attack_name=attack_name,
                                                                 epsilon=epsilon)
            print(adversarial_images.shape)
            confirm_directory(self.keras_model.path+'adversarial_images/'+save_dir)
            np.save(self.keras_model.path+'adversarial_images/'+save_name, adversarial_images)
        self.evaluate_quantized_model_accuracy(adversarial_images)