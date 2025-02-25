import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def CustomModel():
    custom_model = Sequential()

    # First Layer
    custom_model.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=(400, 400, 3)))
    custom_model.add(layers.BatchNormalization())  # Add Batch Normalization
    custom_model.add(layers.MaxPooling2D((2, 2)))

    # Second Layer
    custom_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    custom_model.add(layers.BatchNormalization())  # Add Batch Normalization
    custom_model.add(layers.MaxPooling2D((2, 2)))

    # Last Layer
    custom_model.add(layers.Flatten())
    custom_model.add(layers.Dropout(0.5))  # Add Dropout

    custom_model.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)))  # L2 Regularization
    custom_model.add(layers.Dropout(0.5))  # Add Dropout

    custom_model.add(layers.Dense(1, activation='sigmoid'))

    #custom_model.summary()

    return custom_model


def ResNetModel(num_layers):
    pretrained_model = tf.keras.applications.ResNet50(
        include_top=False,
        input_shape=(400, 400, 3),
        classes=2,
        weights='imagenet',
    )

    for layer in pretrained_model.layers[:num_layers]:
        layer.trainable = False

    #pretrained_model.trainable = False  # Freeze ResNet layers initially

    # num_layers = len(pretrained_model.layers)
    # print(f"Total number of layers in ResNet50: {num_layers}")

    model = pretrained_model.output
    model = layers.GlobalAveragePooling2D()(model) #GAP Layer

    model = layers.BatchNormalization()(model)
    model = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(model) #L2 Regularization added, can be adjusted or removed
    model = layers.Dropout(0.5)(model)

    model = layers.BatchNormalization()(model)
    model = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(model) #L2 Regularization added, can be adjusted or removed
    model = layers.Dropout(0.5)(model)

    predictions = layers.Dense(1, activation='sigmoid')(model) # 2 output classes

    resnet_model = keras.Model(inputs=pretrained_model.input, outputs=predictions)

    # Add the layers
    # model.add(pretrained_model)
    # model.add(layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    # model.add(layers.Dropout(0.3))
    #model.add(layers.BatchNormalization())

    # model.add(layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001))) # Added Dense Layer
    # model.add(layers.Dropout(0.3))
    # # model.add(layers.BatchNormalization())

    # the last layer must specify how many number classes needed to evaluate
    #model.add(layers.Dense(1, activation = 'sigmoid'))

    #model.summary()

    return resnet_model