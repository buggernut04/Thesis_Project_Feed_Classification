import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def CustomModel():
    custom_model = Sequential()

    # Convolutional Layers
    custom_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(500, 300, 3))) 
    custom_model.add(MaxPooling2D((2, 2)))
    custom_model.add(Conv2D(64, (3, 3), activation='relu'))
    custom_model.add(MaxPooling2D((2, 2)))
    custom_model.add(Conv2D(128, (3, 3), activation='relu'))
    custom_model.add(MaxPooling2D((2, 2)))

    # Flatten the output
    custom_model.add(Flatten())

    # Dense Layers
    custom_model.add(Dense(128, activation='relu'))
    custom_model.add(Dense(64, activation='relu'))

    custom_model.add(Dense(2, activation='softmax')) 

    custom_model.summary() 

    return custom_model


def ResNetModel():
    resnet_model = Sequential()

    pretrained_model = tf.keras.applications.ResNet50(
        include_top=False,
        input_shape=(500, 500, 3),
        pooling = 'avg',
        classes=2,
        weights=None,
    )

    # Add the layers
    resnet_model.add(pretrained_model)
    resnet_model.add(layers.Flatten())
    resnet_model.add(layers.Dense(512, activation = 'relu'))
    resnet_model.add(layers.Dropout(0.5))

    # the last layer must specify how many number classes needed to evaluate
    resnet_model.add(layers.Dense(1, activation = 'sigmoid'))

    resnet_model.summary()

    return resnet_model