import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import sys
# import matplotlib.pyplot as plt

from art import config
from art.estimators.classification import KerasClassifier, TensorFlowV2Classifier
from art.attacks.evasion import ProjectedGradientDescent, CarliniL2Method, AutoAttack, AutoProjectedGradientDescent, DeepFool, SquareAttack
from art.utils import get_file



class Attack:
    def __init__(self, data, model):

        self.data = data
        self.attack_dictionary = {
            'cw_official': self.evaluate_on_cw_official,
            'pgd': self.evaluate_on_pgd,
            'auto': self.evaluate_on_auto_attack
        }
        self.num_classes = self.data.y_test.shape[1]
        self.model = model
    

    def evaluate_on_pgd(self, x_input, y_input, epsilon,
                        max_iter=1000, targeted=False, verbose=False):

        loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        # loss_object = tf.keras.losses.CategoricalCrossentropy()
        classifier = TensorFlowV2Classifier(model=self.model.model, nb_classes=self.num_classes, 
                                            input_shape=x_input.shape[1:], loss_object=loss_object,
                                            clip_values=(0, 1), channels_first=False)
        # classifier = KerasClassifier(model=self.model.model, use_logits=False, clip_values=[0,1])
        y_pred = classifier.predict(x_input)
        
        attack = ProjectedGradientDescent(classifier, eps=epsilon, eps_step=0.01,
                                          max_iter=max_iter, targeted=targeted, 
                                          num_random_init=False, verbose=False)
        x_adv = attack.generate(x_input, y=y_input)
        y_adv = classifier.predict(x_adv)
        
        if verbose:
            acc_b = self.model.model.evaluate(x_input, y_input)[1]
            acc_a = self.model.model.evaluate(x_input, y_adv)[1]
            print("Accuracy before the PGD attack: ", acc_b)
            print("Accuracy after the PGD attack:", acc_a)

        return x_adv
        

    def evaluate_on_cw_official(self, x_input, y_input, epsilon,
                                max_iter=1000, targeted=False, verbose=False):
        
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy() 
        classifier = TensorFlowV2Classifier(model=self.model.logits_model, nb_classes=self.num_classes, 
                                            input_shape=x_input.shape[1:], loss_object=loss_object,
                                            clip_values=(0, 1), channels_first=False)
        n = 100
        y_pred = classifier.predict(self.data.x_test[:n])

        attack = CarliniL2Method(classifier, confidence=1.0, max_iter=max_iter, 
                                 batch_size=n, targeted=targeted, verbose=False)
        n = 100
        x_adv = attack.generate(self.data.x_test[:n], y=self.data.y_test[:n])
        y_adv = classifier.predict(x_adv)
        
        if verbose:
            acc_b = np.mean(np.argmax(y_pred, axis=1) == np.argmax(self.data.y_test[:n], axis=1))
            acc_a = np.mean(np.argmax(y_adv, axis=1) == np.argmax(self.data.y_test[:n], axis=1))
            print("Accuracy before the CW-official attack: ", acc_b)
            print("Accuracy after the CW-official attack:", acc_a)

        return x_adv


    def evaluate_on_auto_attack(self, x_input, y_input, epsilon,
                                max_iter=100, targeted=False, verbose=False):
        
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy() 
        classifier = TensorFlowV2Classifier(model=self.model.model, nb_classes=self.num_classes, 
                                            input_shape=x_input.shape[1:], loss_object=loss_object,
                                            clip_values=(0, 1), channels_first=False)
        n = 100
        y_pred = classifier.predict(self.data.x_test[:n])
        
        attacks_list = []
        attacks_list.append(AutoProjectedGradientDescent(classifier, max_iter=max_iter, eps=epsilon, 
                                                         batch_size=n, targeted=targeted, verbose=False,
                                                         loss_type='cross_entropy'))
        attacks_list.append(DeepFool(classifier, batch_size=n, verbose=False))
        attacks_list.append(SquareAttack(classifier, eps=epsilon, max_iter=1000, verbose=False))

        attack = AutoAttack(classifier, eps=epsilon, batch_size=n, 
                            targeted=targeted, attacks=attacks_list)
        n = 100
        x_adv = attack.generate(self.data.x_test[:n], y=self.data.y_test[:n])
        y_adv = classifier.predict(x_adv)
        
        if verbose:
            acc_b = np.sum(np.argmax(y_pred, axis=1) == np.argmax(self.data.y_test[:n], axis=1))
            acc_a = np.sum(np.argmax(y_adv, axis=1) == np.argmax(self.data.y_test[:n], axis=1))
            print("Accuracy before the Auto attack: ", acc_b)
            print("Accuracy after the Auto attack:", acc_a)

        return x_adv
        
        
    def evaluate_on_attack(self, N, epsilon, attack_name='pgd', targeted=False, verbose=True):
        
        print("Evaluating against the "+ attack_name + " attack.")
        x_adv = self.attack_dictionary[attack_name](self.data.x_test[:N], self.data.y_test[:N],
                                               epsilon, targeted=targeted, verbose=verbose)
        y_adv = self.model.model.predict(x_adv)
        return x_adv
        
    
    def typical_pipeline(self, epsilon=0.2, targeted=False):
        self.evaluate_on_attack(100, epsilon, attack_name='pgd', targeted=targeted, verbose=True)
        self.evaluate_on_attack(100, epsilon, attack_name='cw_official', targeted=targeted, verbose=True)
        self.evaluate_on_attack(100, epsilon, attack_name='auto', targeted=targeted, verbose=True)