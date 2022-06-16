import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv1D, BatchNormalization, Dropout, Lambda, Activation, GlobalAveragePooling1D, Dense, LSTM
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam

### Define some default architectures
LAYERS_3 = [{'f': 128, 'k': 8, 's': 1},
            {'f': 256, 'k': 5, 's': 1},
            {'f': 128, 'k': 3, 's': 1}]




def spars_cnn(width=200, n_vars=112,  layers=LAYERS_3,
              dropout=0.3, n_classes=8, learning_rate=0.0001):
    """ Return a CNN model with the architecture used in the SPARS-LBP project.
    Arguments
    ---------
    width : int, default=200
        The segment width of the input data, in number of samples per segment.
    n_vars : int, default=112
        The number of input channels for the model. The default number represents
        the full set of IMU sensors (acceleration, gyroscope, magnetometer,
        pressure, and quaternions) with the full set of 8 IMUs.
    layers : list of dict
        The architecture of the CNN, represented as a list where each element is
        a dictionary containing the hyperparameters (input features, kernel
        size, stride) of each convolutional layer.
    dropout : float, default= 0.3
        The dropout to apply after each convolutional layer.
    n_classes : int, default=8
        The number of output classes for the network. By default this is 8,
        which can be used for exercise classification in the LBP study.
    learning_rate : float, default = 0.0001
        The learning rate of the optimizer.

    Returns:
    --------
    model : tensorflow.Keras.Model
        The initialized CNN model object, ready to be trained.
    """

    # Define the input shape and data type to the model
    input_shape = (width, n_vars)
    model = Sequential()
    input_layer = Input(shape=input_shape)
    model.add(input_layer)

    # For each layer in the model, add a 1D conv, batch norm, and relu layer
    for lp in layers:
        model.add(Conv1D(filters=lp['f'], kernel_size=lp['k'], strides=lp['s'], padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        if dropout:
            model.add(Dropout(rate=dropout))

    # After convolutional layers add global avg pooling and L2 norm
    model.add(GlobalAveragePooling1D())
    model.add(Lambda(lambda x: K.l2_normalize(x, axis=1)))

    # Now add fully-connected layers
    model.add(Dense(units=n_classes, activation='softmax'))

    # Define optimizer and loss metrics
    optimizer = Adam(learning_rate, clipnorm=1.0, amsgrad=True)
    model.compile(loss='categorical_crossentropy',
                       optimizer=optimizer, metrics=['accuracy'])
    return model
