from tensorflow.keras import Sequential, layers
import tensorflow as tf



def cnn_by_atif(data):
    model = Sequential ([
        layers.Conv2D(filters=8, input_shape=data.get_input_shape(), 
                      kernel_size=(3,3), strides=(1,1), padding='same',
                      activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
        
        layers.Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding='same',
                      activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
        
        layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='same',
                      activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
        
        layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same',
                      activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
        
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(data.get_output_shape()[0], name="logits_layer"),
        layers.Activation('softmax')
    ])
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.summary()
    return model
    
    

def cnn_by_atif_without_activation(data):
    model = Sequential ([
        layers.Conv2D(filters=8, input_shape=data.get_input_shape(), 
                      kernel_size=(3,3), strides=(2,2), padding='same'),
        layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
        
        layers.Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding='same'),
        layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
        
        layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='same'),
        layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
        
        layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same'),
        layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
        
        layers.Flatten(),
        layers.Dense(512),
        layers.Dense(data.get_output_shape()[0], name="logits_layer"),
        layers.Activation('softmax')
    ])
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.summary()
    return model



def cnn_by_hassan(data):
    model = Sequential ([
        layers.Conv2D(filters=8, input_shape=data.get_input_shape(), 
                      kernel_size=(3,3), strides=(2,2), padding='same',
                      activation='relu'),
        
        layers.Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding='same',
                      activation='relu'),
        
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(data.get_output_shape()[0], name="logits_layer"),
        layers.Activation('softmax')
    ])
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.summary()
    return model
    

def ae_by_hassan(data):
    model = Sequential ([
        layers.Conv2D(filters=8, input_shape=data.get_input_shape(), 
                      kernel_size=(3,3), strides=(2,2), padding='same',
                      activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
        
        layers.UpSampling2D(size=(4,4)),
        
        layers.Conv2D(filters=3, kernel_size=(3,3), strides=(1,1), padding='same',
                      activation='relu')
    ])
    
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    model.summary()
    return model


def ae_by_hassan_gray(data):
    model = Sequential ([
        layers.Conv2D(filters=8, input_shape=data.x_test.shape[1:-1]+tuple([1]), 
                      kernel_size=(3,3), strides=(2,2), padding='same',
                      activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
        
        layers.UpSampling2D(size=(4,4)),
        
        layers.Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding='same',
                      activation='relu')
    ])
    
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    model.summary()
    return model


def quantized_model(data, keras_model, quantization_levels=16):
    from utils.custom_layer_utils.input_processing import RGB2Gray3, RGB2Gray, Gray2Gray3
    from utils.custom_layer_utils.quantization import Quantization, Random_Quantization
    
    model_gray = Sequential()
    model_gray.add(layers.Input(shape=data.get_input_shape()))
    # model_gray.add(RGB2Gray())
    model_gray.add(keras_model.auto_encoder)
    # model_gray.add(Gray2Gray3())
    model_gray.add(Quantization(quantization_levels=quantization_levels))
    model_gray.add(keras_model.model)
    
    model_gray.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model_gray


def cnn_by_atif_decoder(data):
    decoder_layers = [
        layers.UpSampling2D(size=(2,2), input_shape=(8,8,64)),
        layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='same',
                              activation='relu'),
        layers.UpSampling2D(size=(2,2), input_shape=(16,16,32)),
        layers.Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding='same',
                              activation='relu'),
        layers.UpSampling2D(size=(2,2), input_shape=(32,32,16)),
        layers.Conv2D(filters=8, kernel_size=(3,3), strides=(1,1), padding='same',
                              activation='relu'),
        layers.UpSampling2D(size=(2,2), input_shape=(64,64,8)),
        layers.Conv2D(filters=3, kernel_size=(3,3), strides=(1,1), padding='same',
                              activation='relu')
    ]
    return decoder_layers
    

def auto_encoded_model_by_atif(data, encoder_layers=7):
    model = cnn_by_atif(data)
    encoder = Sequential(name='encoder')
    for layer in model.layers[:encoder_layers]:
        encoder.add(layer)
    encoder.compile(loss='mse', optimizer='adam', metrics=['mse'])
    # encoder.summary()
    
    decoder_layers = cnn_by_atif_decoder(data)
    decoder = Sequential(name='decoder')
    for layer in decoder_layers[-encoder_layers:]:
        decoder.add(layer)
    decoder.compile(loss='mse', optimizer='adam', metrics=['mse'])
    # decoder.summary()
    
    
    classifier = Sequential(name='classifier')
    classifier.add(layers.Flatten(input_shape=encoder.output_shape[1:]))
    classifier.add(layers.Dense(512, activation='relu'))
    classifier.add(layers.Dense(data.get_output_shape()[0], name="logits_layer"))
    classifier.add(layers.Activation('softmax'))
    classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    model = Sequential(name='model')
    model.add(encoder)
    model.add(classifier)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    auto_encoder = Sequential(name='auto_encoder')
    auto_encoder.add(encoder)
    auto_encoder.add(decoder)
    auto_encoder.compile(loss='mse', optimizer='adam', metrics=['mse'])
    
    encoder_in = layers.Input(shape=data.get_input_shape())
    encoder_out = encoder(encoder_in)
    decoder_out = decoder(encoder_out)
    classifier_out = classifier(encoder_out)
    auto_encoded_model = tf.keras.models.Model(encoder_in, [classifier_out, decoder_out])
    auto_encoded_model.compile(loss=['categorical_crossentropy', 'mse'], optimizer='adam', metrics=['mse'])
    auto_encoded_model.summary()
    return auto_encoded_model, auto_encoder, model


def auto_encoded_model_by_atif_definition(data):
    encoder = Sequential()
    encoder.add(layers.Conv2D(filters=8, input_shape=data.get_input_shape(), 
                              kernel_size=(3,3), strides=(1,1), padding='same',
                              activation='relu'))
    encoder.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))
    encoder.add(layers.Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding='same',
                              activation='relu'))
    encoder.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))
    encoder.add(layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='same',
                              activation='relu'))
    encoder.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))
    encoder.add(layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same',
                              activation='relu'))
    encoder.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))
    encoder.compile(loss='mse', optimizer='adam', metrics=['mse'])
    # encoder.summary()
    
    decoder=Sequential()
    decoder.add(layers.UpSampling2D(size=(4,4), input_shape=(4,4,64)))
    decoder.add(layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='same',
                              activation='relu'))
    decoder.add(layers.UpSampling2D(size=(4,4)))
    decoder.add(layers.Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding='same',
                              activation='relu'))
    decoder.add(layers.UpSampling2D(size=(4,4)))
    decoder.add(layers.Conv2D(filters=3, kernel_size=(3,3), strides=(2,2), padding='same',
                              activation='relu'))
    decoder.compile(loss='mse', optimizer='adam', metrics=['mse'])
    # decoder.summary()
    
    
    classifier = Sequential()
    classifier.add(layers.Flatten(input_shape=encoder.output_shape[1:]))
    classifier.add(layers.Dense(512, activation='relu'))
    classifier.add(layers.Dense(data.get_output_shape()[0], name="logits_layer"))
    classifier.add(layers.Activation('softmax'))
    
    model = Sequential()
    model.add(encoder)
    model.add(classifier)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    auto_encoder = Sequential()
    auto_encoder.add(encoder)
    auto_encoder.add(decoder)
    auto_encoder.compile(loss='mse', optimizer='adam', metrics=['mse'])
    
    encoder_in = layers.Input(shape=data.get_input_shape())
    encoder_out = encoder(encoder_in)
    decoder_out = decoder(encoder_out)
    classifier_out = classifier(encoder_out)
    auto_encoded_model = tf.keras.models.Model(encoder_in, [classifier_out, decoder_out])
    auto_encoded_model.compile(loss=['categorical_crossentropy', 'mse'], optimizer='adam', metrics=['mse'])
    auto_encoded_model.summary()
    return auto_encoded_model, auto_encoder, model