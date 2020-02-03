from keras import Model
from keras.layers import Conv1D, Input, BatchNormalization, Dropout, Lambda, Activation, GlobalAveragePooling1D
import keras.backend as K


def fcn(input_shape, conv_layers=({'f': 128, 'k': 8, 's': 1}, {'f': 256, 'k': 5, 's': 1}, {'f': 128, 'k': 3, 's': 1}),
        dropout=0.3, normalize=True, embedding_size=None):
    """
    Creates fully convolutional neural (FCN) network architecture described in: https://arxiv.org/abs/2001.05517

    :param input_shape: tuple (2)
        segment shape (width, n_channels)
    :param conv_layers: tuple of dicts
        describe conv layers with f: filters, k: kernel size, s: stride
    :param dropout: float
        dropout ratio applied at each layer
    :param normalize: bool
        apply l2 normalization
    :param embedding_size: integer, optional
        defines embedding size (number of filters for last CNN layer)
    :return: keras model
        the fcn model
    """

    input_layer = Input(shape=input_shape)
    layer = input_layer

    if embedding_size:
        conv_layers[-1]['f'] = embedding_size

    for lp in conv_layers:
        layer = Conv1D(filters=lp['f'], kernel_size=lp['k'], strides=lp['s'], padding='same')(layer)
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)

        if dropout:
            layer = Dropout(rate=dropout)(layer)

    layer = GlobalAveragePooling1D()(layer)

    if normalize:
        layer = Lambda(lambda x: K.l2_normalize(x, axis=1))(layer)

    return Model(inputs=input_layer, outputs=layer)
