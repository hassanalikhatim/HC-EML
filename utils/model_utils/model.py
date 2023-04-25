import tensorflow as tf
import os
import numpy as np

from utils.model_utils.model_architectures import ae_by_hassan, ae_by_hassan_gray, auto_encoded_model_by_atif
from utils.attack_utils.attack import Attack
from utils.custom_layer_utils.quantization import Quantization



def confirm_directory(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return

class Keras_Model:
    def __init__(self, path, data=None, model=None, model_architecture=None, 
                 auto_encoded_model=None, verbose=True):
        self.data = data
        self.model = model
        self.path = path
        self.model_architecture = model_architecture
        self.auto_encoded_model = auto_encoded_model
        self.verbose = verbose
    
    
    def print_out(self, *print_statement, end=""):
        if self.verbose:
            print(*print_statement, end=end)
    
        
    def prepare_logits_model(self):
        self.logits_model = tf.keras.models.Model(inputs=self.model.input, 
                                                  outputs=self.model.get_layer("logits_layer").output)
        
    
    def train_and_save(self, epochs=3, batch_size=64, model_name=None):
        self.model.fit(self.data.x_train, self.data.y_train,
                       epochs=epochs, batch_size=batch_size,
                       validation_data=(self.data.x_test, self.data.y_test))
        if model_name is not None:
            self.model_name = model_name
        else:
            self.model_name = self.data.data_name
        self.model.save(self.path+'models/'+self.model_name+'.h5')
        
        
    def train_on_batch_and_save(self, epochs=3, batch_size=64, model_name=None,
                                total_batches=6):
        for epoch in range(epochs):
            for batch_number in range(total_batches):
                self.data.load_train_batch(batch_number)
                self.model.fit(self.data.x_train_batch, self.data.y_train_batch,
                               epochs=1, batch_size=batch_size,
                               validation_data=(self.data.x_test, self.data.y_test))
                del self.data.x_train_batch
                del self.data.y_train_batch
            
        if model_name is not None:
            self.model_name = model_name
        else:
            self.model_name = self.data.data_name
        self.model.save(self.path+'models/'+self.model_name+'.h5')
                
                
    def compute_adversarial_images(self, N=100, attack_name='pgd',
                                      epsilon=0.01):
        attack = Attack(self.data, self)
        adversarial_images = attack.evaluate_on_attack(N, epsilon, attack_name=attack_name)
        self.model.evaluate(adversarial_images, self.data.y_test[:N])
        return adversarial_images
    
        
    def evaluate_quantized_model_accuracy(self, adversarial_images,
                                          quantization_levels=([2, 4, 8, 16, 255])):
        clean_accuracies = []
        adv_accuracies = []
        for quantization_level in quantization_levels:
            quantized_model = tf.keras.Sequential()
            quantized_model.add(tf.keras.layers.Input(shape=self.data.get_input_shape()))
            quantized_model.add(Quantization(quantization_levels=quantization_level))
            quantized_model.add(self.model)
            quantized_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            clean_acc = quantized_model.evaluate(self.data.x_test, 
                                                 self.data.y_test, 
                                                 verbose=False)[1]
            adv_acc = quantized_model.evaluate(adversarial_images, 
                                               self.data.y_test[:len(adversarial_images)], 
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
        save_dir = self.model_name+'/'
        save_name = save_dir+'adversarial_images_'+attack_name+'('+str(epsilon)+')'
        try:
            adversarial_images = np.load(self.path+'adversarial_images/'+save_name+'.npy')
            print(adversarial_images.shape)
        except:
            adversarial_images = self.compute_adversarial_images(attack_name=attack_name,
                                                                 epsilon=epsilon)
            print(adversarial_images.shape)
            confirm_directory(self.path+'adversarial_images/'+save_dir)
            np.save(self.path+'adversarial_images/'+save_name, adversarial_images)
        self.evaluate_quantized_model_accuracy(adversarial_images)
        
        
    def train_autoencoded_model_and_save(self, epochs=3, batch_size=20, total_batches=6,
                                         encoder_layers=8):
        auto_encoded_model, auto_encoder, model = auto_encoded_model_by_atif(self.data, 
                                                                             encoder_layers=encoder_layers)
        try:
            auto_encoded_model.load_weights('autoencoded_model_'+str(encoder_layers)+'.h5')
            model.evaluate(self.data.x_test, self.data.y_test)
        except:
            print("Training from scratch")
        for epoch in range(epochs):
            for batch_number in range(total_batches):
                self.data.load_train_batch(batch_number)
                auto_encoded_model.fit(
                    self.data.x_train_batch, 
                    [self.data.y_train_batch, self.data.x_train_batch],
                    epochs=1, batch_size=batch_size, shuffle=True,
                    validation_data=(
                        self.data.x_test,
                        [self.data.y_test, self.data.x_test]
                        ),
                    verbose=False
                    )
                del self.data.x_train_batch
                del self.data.y_train_batch
                auto_encoded_model.save_weights('autoencoded_model_'+str(encoder_layers)+'.h5')
            ae_loss = auto_encoder.evaluate(self.data.x_test, self.data.x_test, verbose=False)[1]
            m_acc = model.evaluate(self.data.x_test, self.data.y_test, verbose=False)[1]
            print(epoch, "AE loss: ", ae_loss, ", \tModel accuracy: ", m_acc)