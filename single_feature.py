import tensorflow as tf
import numpy as np

import util

class PhonemeEmbeddedLayer(tf.keras.layers.Layer):
    def __init__(self, units_embedded=64, units_lstm=32):
        # layer
        self.max_pool = tf.keras.layers.MaxPool2D(pool_size=(3, 1), strides=(3, 1), padding='same')

        self.layer_exp1 = tf.keras.layers.Lambda(lambda x: tf.keras.backend.expand_dims(x, axis=3))
        self.embedding_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units_embedded, use_bias=False))

        # self.lstm_layer = tf.keras.layers.LSTM(units_lstm, return_sequences=True)
        self.lstm_layer = tf.keras.layers.CuDNNLSTM(units_lstm, return_sequences=True)

        self.squeeze_layer = util.SqueezeLayer()


    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'units_embedded': self.units_embedded,
            'units_lstm': self.units_lstm,
        })
        return config


    def __call__(self, input_phoneme):
        phoneme_proj = input_phoneme

        phoneme_out = self.layer_exp1(phoneme_proj)
        # layer
        phoneme_out = self.max_pool(phoneme_out)

        phoneme_out = self.squeeze_layer(phoneme_out)
        # layer
        phoneme_out = self.embedding_layer(phoneme_out)
        # layer

        phoneme_out = self.lstm_layer(phoneme_out)  # size = (210,16)


        return phoneme_out

# LSTM-based model which uses mel spectrogram as speech representation in match/mismatch task
def lstm_mel(shape_eeg, shape_spch, units_lstm=32, filters_cnn_eeg=16, filters_cnn_env=16,
                            units_hidden=128,
                            stride_temporal=3, kerSize_temporal=9, spatial_filters_eeg=32,
                            spatial_filters_mel=8, fun_act='tanh'):
    """
    Return an LSTM based model where batch normalization is applied to input of each layer.

    :param shape_eeg: a numpy array, shape of EEG signal (time, channel)
    :param shape_spch: a numpy array, shape of speech signal (time, feature_dim)
    :param units_lstm: an int, number of units in LSTM
    :param filters_cnn_eeg: an int, number of CNN filters applied on EEG
    :param filters_cnn_env: an int, number of CNN filters applied on envelope
    :param units_hidden: an int, number of units in the first time_distributed layer
    :param stride_temporal: an int, amount of stride in the temporal direction
    :param kerSize_temporal: an int, size of CNN filter kernel in the temporal direction
    :param fun_act: activation function used in layers
    :return: LSTM-based model
    """

    ############
    input_eeg = tf.keras.layers.Input(shape=shape_eeg)
    input_spch1 = tf.keras.layers.Input(shape=shape_spch)
    input_spch2 = tf.keras.layers.Input(shape=shape_spch)

    ############
    #### upper part of network dealing with EEG.

    layer_exp1 = tf.keras.layers.Lambda(lambda x: tf.keras.backend.expand_dims(x, axis=3))
    eeg_proj = input_eeg

    # layer
    output_eeg = tf.keras.layers.BatchNormalization()(eeg_proj)  # batch normalization
    output_eeg = tf.keras.layers.Conv1D(spatial_filters_eeg, kernel_size=1)(output_eeg)

    # layer
    output_eeg = tf.keras.layers.BatchNormalization()(output_eeg)
    output_eeg = layer_exp1(output_eeg)
    output_eeg = tf.keras.layers.Convolution2D(filters_cnn_eeg, (kerSize_temporal, 1),
                                               strides=(stride_temporal, 1), activation="relu")(output_eeg)

    # layer
    layer_permute = tf.keras.layers.Permute((1, 3, 2))
    output_eeg = layer_permute(output_eeg)

    layer_reshape = tf.keras.layers.Reshape((tf.keras.backend.int_shape(output_eeg)[1],
                                             tf.keras.backend.int_shape(output_eeg)[2] *
                                             tf.keras.backend.int_shape(output_eeg)[3]))
    output_eeg = layer_reshape(output_eeg)

    layer2_timeDis = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units_hidden, activation=fun_act))
    output_eeg = layer2_timeDis(output_eeg)

    # layer
    output_eeg = tf.keras.layers.BatchNormalization()(output_eeg)
    layer3_timeDis = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units_lstm, activation=fun_act))
    output_eeg = layer3_timeDis(output_eeg)

    ##############
    #### Bottom part of the network dealing with Speech.

    spch1_proj = input_spch1
    spch2_proj = input_spch2

    # layer
    BN_layer = tf.keras.layers.BatchNormalization()
    output_spch1 = BN_layer(spch1_proj)
    output_spch2 = BN_layer(spch2_proj)

    env_spatial_layer = tf.keras.layers.Conv1D(spatial_filters_mel, kernel_size=1)
    output_spch1 = env_spatial_layer(output_spch1)
    output_spch2 = env_spatial_layer(output_spch2)

    # layer
    BN_layer1 = tf.keras.layers.BatchNormalization()
    output_spch1 = BN_layer1(output_spch1)
    output_spch2 = BN_layer1(output_spch2)

    output_spch1 = layer_exp1(output_spch1)
    output_spch2 = layer_exp1(output_spch2)

    conv_env_layer = tf.keras.layers.Convolution2D(filters_cnn_env, (kerSize_temporal, 1),
                                                   strides=(stride_temporal, 1), activation="relu")
    output_spch1 = conv_env_layer(output_spch1)
    output_spch2 = conv_env_layer(output_spch2)

    # layer
    BN_layer2 = tf.keras.layers.BatchNormalization()
    output_spch1 = BN_layer2(output_spch1)
    output_spch2 = BN_layer2(output_spch2)

    output_spch1 = layer_permute(output_spch1)
    output_spch2 = layer_permute(output_spch2)

    layer_reshape = tf.keras.layers.Reshape((tf.keras.backend.int_shape(output_spch1)[1],
                                             tf.keras.backend.int_shape(output_spch1)[2] *
                                             tf.keras.backend.int_shape(output_spch1)[3]))
    output_spch1 = layer_reshape(output_spch1)  # size = (210,32)
    output_spch2 = layer_reshape(output_spch2)

    # lstm_spch = tf.keras.layers.LSTM(units_lstm, return_sequences=True, activation= fun_act)
    lstm_spch = tf.compat.v1.keras.layers.CuDNNLSTM(units_lstm, return_sequences=True)
    output_spch1 = lstm_spch(output_spch1)
    output_spch2 = lstm_spch(output_spch2)

    ##############
    #### last common layers
    # layer
    layer_dot = util.DotLayer()
    cos_scores = layer_dot([output_eeg, output_spch1])
    cos_scores2 = layer_dot([output_eeg, output_spch2])

    # layer
    layer_expand = tf.keras.layers.Lambda(lambda x: tf.keras.backend.expand_dims(x, axis=2))
    layer_sigmoid = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation='sigmoid'))

    cos_scores_mix = tf.keras.layers.Concatenate()([layer_expand(cos_scores), layer_expand(cos_scores2)])

    cos_scores_sig = layer_sigmoid(cos_scores_mix)

    # layer
    layer_ave = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1, keepdims=True))
    cos_scores_sig = util.SqueezeLayer()(cos_scores_sig, axis=2)
    y_out = layer_ave(cos_scores_sig)

    model = tf.keras.Model(inputs=[input_eeg, input_spch1, input_spch2], outputs=[y_out, cos_scores_sig])

    return model




# LSTM-based model that uses phonemes as speech representation in match/mismatch task
def lstm_phoneme(shape_eeg, shape_spch, units_lstm=32, filters_cnn_eeg=16,
                            units_hidden=128,
                            stride_temporal=3, kerSize_temporal=9, spatial_filters_eeg=32,
                            units_embedded= 64, fun_act='tanh'):

    ############
    input_eeg = tf.keras.layers.Input(shape=shape_eeg)
    phoneme1 = tf.keras.layers.Input(shape=shape_spch)
    phoneme2 = tf.keras.layers.Input(shape=shape_spch)

    ############
    #### upper part of network dealing with EEG.

    layer_exp1 = tf.keras.layers.Lambda(lambda x: tf.keras.backend.expand_dims(x, axis=3))
    eeg_proj = input_eeg

    # layer
    output_eeg = tf.keras.layers.BatchNormalization()(eeg_proj)  # batch normalization
    output_eeg = tf.keras.layers.Conv1D(spatial_filters_eeg, kernel_size=1)(output_eeg)

    # layer
    output_eeg = tf.keras.layers.BatchNormalization()(output_eeg)
    output_eeg = layer_exp1(output_eeg)
    output_eeg = tf.keras.layers.Convolution2D(filters_cnn_eeg, (kerSize_temporal, 1),
                                               strides=(stride_temporal, 1), activation="relu", padding='same')(output_eeg)

    # layer
    layer_permute = tf.keras.layers.Permute((1, 3, 2))
    output_eeg = layer_permute(output_eeg)

    layer_reshape = tf.keras.layers.Reshape((tf.keras.backend.int_shape(output_eeg)[1],
                                             tf.keras.backend.int_shape(output_eeg)[2] *
                                             tf.keras.backend.int_shape(output_eeg)[3]))
    output_eeg = layer_reshape(output_eeg)

    layer2_timeDis = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units_hidden, activation=fun_act))
    output_eeg = layer2_timeDis(output_eeg)

    # layer
    output_eeg = tf.keras.layers.BatchNormalization()(output_eeg)
    layer3_timeDis = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units_lstm, activation=fun_act))
    output_eeg = layer3_timeDis(output_eeg)

    ##############
    #### Bottom part of the network dealing with Envelope.

    phoneme1_proj = phoneme1
    phoneme2_proj = phoneme2

    phoneme_layer = PhonemeEmbeddedLayer(units_embedded=units_embedded, units_lstm=units_lstm)
    phoneme1_out = phoneme_layer(phoneme1_proj)
    phoneme2_out = phoneme_layer(phoneme2_proj)

    ##############
    #### last common layers
    # layer
    layer_dot = util.DotLayer()
    cos_scores = layer_dot([output_eeg, phoneme1_out])
    cos_scores2 = layer_dot([output_eeg, phoneme2_out])

    # layer
    layer_expand = tf.keras.layers.Lambda(lambda x: tf.keras.backend.expand_dims(x, axis=2))
    layer_sigmoid = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation='sigmoid'))

    cos_scores_mix = tf.keras.layers.Concatenate()([layer_expand(cos_scores), layer_expand(cos_scores2)])

    cos_scores_sig = layer_sigmoid(cos_scores_mix)

    # layer
    layer_ave = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1, keepdims=True))
    cos_scores_sig = util.SqueezeLayer()(cos_scores_sig, axis=2)
    y_out = layer_ave(cos_scores_sig)

    model = tf.keras.Model(inputs=[input_eeg, phoneme1, phoneme2], outputs=[y_out, cos_scores_sig])

    return model




# LSTM-based model which uses word embedding as speech representation in match/mismatch task
def lstm_spch_pooling(shape_eeg, shape_spch, units_lstm=32, filters_cnn_eeg=16, filters_cnn_env=16,
                            units_hidden=128,
                            stride_temporal=3, kerSize_temporal=9, spatial_filters_eeg=32, fun_act='tanh'):

    ############
    input_eeg = tf.keras.layers.Input(shape=shape_eeg)
    input_spch1 = tf.keras.layers.Input(shape=shape_spch)
    input_spch2 = tf.keras.layers.Input(shape=shape_spch)

    ############
    #### upper part of network dealing with EEG.

    layer_exp1 = tf.keras.layers.Lambda(lambda x: tf.keras.backend.expand_dims(x, axis=3))
    eeg_proj = input_eeg

    # layer
    output_eeg = tf.keras.layers.BatchNormalization()(eeg_proj)  # batch normalization
    output_eeg = tf.keras.layers.Conv1D(spatial_filters_eeg, kernel_size=1)(output_eeg)

    # layer
    output_eeg = tf.keras.layers.BatchNormalization()(output_eeg)
    output_eeg = layer_exp1(output_eeg)
    output_eeg = tf.keras.layers.Convolution2D(filters_cnn_eeg, (kerSize_temporal, 1),
                                               strides=(stride_temporal, 1), activation="relu", padding='same')(output_eeg)

    # layer
    layer_permute = tf.keras.layers.Permute((1, 3, 2))
    output_eeg = layer_permute(output_eeg)

    layer_reshape = tf.keras.layers.Reshape((tf.keras.backend.int_shape(output_eeg)[1],
                                             tf.keras.backend.int_shape(output_eeg)[2] *
                                             tf.keras.backend.int_shape(output_eeg)[3]))
    output_eeg = layer_reshape(output_eeg)

    layer2_timeDis = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units_hidden, activation=fun_act))
    output_eeg = layer2_timeDis(output_eeg)

    # layer
    output_eeg = tf.keras.layers.BatchNormalization()(output_eeg)
    layer3_timeDis = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units_lstm, activation=fun_act))
    output_eeg = layer3_timeDis(output_eeg)

    ##############
    #### Bottom part of the network dealing with Speech.

    spch1_proj = input_spch1
    spch2_proj = input_spch2


    # layer

    output_spch1 = layer_exp1(spch1_proj)
    output_spch2 = layer_exp1(spch2_proj)

    max_pool = tf.keras.layers.MaxPool2D(pool_size=(3, 1), strides=(3, 1), padding='same')
    output_spch1 = max_pool(output_spch1)
    output_spch2 = max_pool(output_spch2)

    # layer
    BN_layer2 = tf.keras.layers.BatchNormalization()
    output_spch1 = BN_layer2(output_spch1)
    output_spch2 = BN_layer2(output_spch2)

    output_spch1 = layer_permute(output_spch1)
    output_spch2 = layer_permute(output_spch2)

    layer_reshape = tf.keras.layers.Reshape((tf.keras.backend.int_shape(output_spch1)[1],
                                             tf.keras.backend.int_shape(output_spch1)[2] *
                                             tf.keras.backend.int_shape(output_spch1)[3]))
    output_spch1 = layer_reshape(output_spch1)
    output_spch2 = layer_reshape(output_spch2)

    # lstm_spch = tf.keras.layers.LSTM(units_lstm, return_sequences=True, activation= fun_act)
    lstm_spch = tf.compat.v1.keras.layers.CuDNNLSTM(units_lstm, return_sequences=True)
    output_spch1 = lstm_spch(output_spch1)
    output_spch2 = lstm_spch(output_spch2)

    ##############
    #### last common layers
    # layer
    layer_dot = util.DotLayer()
    cos_scores = layer_dot([output_eeg, output_spch1])
    cos_scores2 = layer_dot([output_eeg, output_spch2])

    # layer
    layer_expand = tf.keras.layers.Lambda(lambda x: tf.keras.backend.expand_dims(x, axis=2))
    layer_sigmoid = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation='sigmoid'))

    cos_scores_mix = tf.keras.layers.Concatenate()([layer_expand(cos_scores), layer_expand(cos_scores2)])

    cos_scores_sig = layer_sigmoid(cos_scores_mix)

    # layer
    layer_ave = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1, keepdims=True))
    cos_scores_sig = util.SqueezeLayer()(cos_scores_sig, axis=2)
    y_out = layer_ave(cos_scores_sig)

    model = tf.keras.Model(inputs=[input_eeg, input_spch1, input_spch2], outputs=[y_out, cos_scores_sig])

    return model
