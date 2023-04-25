from tensorflow.keras import utils
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from PIL import Image



class Dataset:
    def __init__(self, data_name='mnist', preferred_size=None, invert_color_scale_flag=False):
        self.data_name = data_name
        self.preferred_size = preferred_size
        self.invert_color_scale_flag = invert_color_scale_flag
    
    
    def load_data(self):
        data = np.load("Data/Affectnet-Dataset/affectnet_dataset.npz", allow_pickle=True)
        self.x_train = data['arr_0']/255.
        self.y_train = data['arr_1']
        self.x_test = data['arr_2']/255.
        self.y_test = data['arr_3']
        
        # self.x_train, self.y_train = permute(self.x_train, self.y_train)
        self.y_train = utils.to_categorical(self.y_train, np.max(self.y_test)+1)
        self.y_test = utils.to_categorical(self.y_test, np.max(self.y_test)+1)
        print(self.x_train.shape, self.y_train.shape, self.x_test.shape, self.y_test.shape)
        
    
    def load_train_batch(self, batch_number):
        data = np.load("Data/Affectnet-Dataset/affectnet_train_set_"+str(batch_number)+".npz")
        self.x_train_batch = data['arr_0']
        self.y_train_batch = data['arr_1']
        self.y_train_batch = utils.to_categorical(self.y_train_batch, self.y_test.shape[1])
    
        
    def load_data_from_directory(self):
        self.x_train, self.y_train, self.x_test, self.y_test = self.load_and_prepare_data()
        
    
    def load_test_data(self):
        data_ = np.load("Data/Affectnet-Dataset/affectnet_test_set.npz", allow_pickle=True)
        self.x_test = data_['arr_0']
        self.y_test = data_['arr_1']
        self.y_test = utils.to_categorical(self.y_test, np.max(self.y_test)+1)
        
    
    def get_input_shape(self):
        return self.x_test.shape[1:]
    
    
    def get_output_shape(self):
        return self.y_test.shape[1:]
    
    
    def get_class_names(self):
        class_names = {
            "Affectnet-Dataset": ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        }
        return class_names[self.data_name]
    
    
    def load_and_prepare_data(self):
        data_loader = {
            "fmnist": tf.keras.datasets.fashion_mnist.load_data,
            "mnist": tf.keras.datasets.mnist.load_data,
            "cifar10": tf.keras.datasets.cifar10.load_data,
            "Affectnet-Dataset": self.get_data
        }
        (x_train, y_train), (x_test, y_test) = data_loader[self.data_name]()
        
        print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
        x_train = x_train.astype('float32')/255.
        x_test = x_test.astype('float32')/255.
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2], -1))
        x_test = np.reshape(x_test, (x_test.shape[0], x_train.shape[1], x_train.shape[2], -1))
        
        num_classes = np.max(y_train) + 1
        y_train = utils.to_categorical(y_train, num_classes)
        y_test = utils.to_categorical(y_test, num_classes)
        
        return x_train, y_train, x_test, y_test
        
    
    def get_data(self):
        directories = ['/Train/', '/Test/']#, '/Validation/']
        class_names = os.listdir('Data/'+self.data_name+'/Train/')

        inputs = []
        outputs = []
        for directory in directories:
            images = []
            labels = []
            data_path = 'Data/'+self.data_name+directory
            for class_num,class_name in enumerate(class_names):
                for file_name in os.listdir(data_path+class_name):
                    image_ = Image.open(data_path+class_name+'/'+file_name).convert('RGB')
                    if self.preferred_size is not None:
                        image_ = image_.resize(self.preferred_size, Image.ANTIALIAS)
                    image_ = np.array(image_)
                    images.append(image_*(255/np.max(image_)))
                    labels.append(class_num)
            labels = np.array(labels)
            inputs.append(np.array(images))
            outputs.append(np.array(labels))

        return (inputs[0], outputs[0]), (inputs[1], outputs[1])