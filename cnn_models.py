import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def CustomModel():
    custom_model = Sequential()

    # First Layer
    custom_model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(400, 400, 3)))
    custom_model.add(layers.BatchNormalization())  # Add Batch Normalization
    custom_model.add(layers.MaxPooling2D((2, 2)))

    # Second Layer
    custom_model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    custom_model.add(layers.BatchNormalization())  # Add Batch Normalization
    custom_model.add(layers.MaxPooling2D((2, 2)))

    # Third Layer
    custom_model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    custom_model.add(layers.BatchNormalization())  # Add Batch Normalization
    custom_model.add(layers.MaxPooling2D((2, 2)))

    # Last Layer
    custom_model.add(layers.Flatten())
    custom_model.add(layers.Dropout(0.5))  # Add Dropout

    custom_model.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)))  # L2 Regularization
    custom_model.add(layers.Dropout(0.5))  # Add Dropout

    custom_model.add(layers.Dense(1, activation='sigmoid'))

    return custom_model


def ResNetModel():
    resnet_model = Sequential()

    pretrained_model = tf.keras.applications.ResNet50(
        include_top=False,
        input_shape=(400, 400, 3),
        pooling = 'avg',
        classes=2,
        weights='imagenet',
    )

    pretrained_model.trainable = False  # Freeze ResNet layers initially

    num_layers = len(pretrained_model.layers)
    print(f"Total number of layers in ResNet50: {num_layers}")

    # Add the layers
    resnet_model.add(pretrained_model)
    resnet_model.add(layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)))  # L2 Regularization
    resnet_model.add(layers.Dropout(0.3))
    resnet_model.add(layers.BatchNormalization())

    resnet_model.add(layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001))) # Added Dense Layer
    resnet_model.add(layers.Dropout(0.3))
    resnet_model.add(layers.BatchNormalization())

    # the last layer must specify how many number classes needed to evaluate
    resnet_model.add(layers.Dense(1, activation = 'sigmoid'))

    resnet_model.summary()

    return resnet_model