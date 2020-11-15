import tensorflow as tf
import numpy as np

import util


# concatenating envelope and mel spectrogram features together
def lstm_env_mel_spatial_filter(shape_eeg, shape_feature1, shape_feature2, units_lstm=32, filters_cnn_eeg=16, filters_cnn_env=16,
                            units_hidden=128,
                            stride_temporal=3, kerSize_temporal=9, spatial_filters_eeg=32,
                            spatial_filters_mel=8, fun_act='tanh'):

    ############
    input_eeg = tf.keras.layers.Input(shape=shape_eeg)
    env1 = tf.keras.layers.Input(shape=shape_feature1)
    env2 = tf.keras.layers.Input(shape=shape_feature1)
    mel1 = tf.keras.layers.Input(shape=shape_feature2)
    mel2 = tf.keras.layers.Input(shape=shape_feature2)

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


    env1_proj = env1
    env2_proj = env2

    # layer
    BN_layer = tf.keras.layers.BatchNormalization()
    output_env1 = BN_layer(env1_proj)
    output_env2 = BN_layer(env2_proj)

    output_env1 = layer_exp1(output_env1)
    output_env2 = layer_exp1(output_env2)

    conv_env_layer = tf.keras.layers.Convolution2D(filters_cnn_env, (kerSize_temporal, 1),
                                                   strides=(stride_temporal, 1), activation="relu")
    output_env1 = conv_env_layer(output_env1)
    output_env2 = conv_env_layer(output_env2)




    ## speech feature 2

    mel1_proj = mel1
    mel2_proj = mel2

    # layer
    BN_layer = tf.keras.layers.BatchNormalization()
    output_mel1 = BN_layer(mel1_proj)
    output_mel2 = BN_layer(mel2_proj)

    env_spatial_layer = tf.keras.layers.Conv1D(spatial_filters_mel, kernel_size=1)
    output_mel1 = env_spatial_layer(output_mel1)
    output_mel2 = env_spatial_layer(output_mel2)

    # layer
    BN_layer1 = tf.keras.layers.BatchNormalization()
    output_mel1 = BN_layer1(output_mel1)
    output_mel2 = BN_layer1(output_mel2)

    output_mel1 = layer_exp1(output_mel1)
    output_mel2 = layer_exp1(output_mel2)

    conv_env_layer = tf.keras.layers.Convolution2D(filters_cnn_env, (kerSize_temporal, 1),
                                                   strides=(stride_temporal, 1), activation="relu")
    output_mel1 = conv_env_layer(output_mel1)
    output_mel2 = conv_env_layer(output_mel2)



    # layer: combine two features

    layer_permute = tf.keras.layers.Permute((1, 3, 2))
    output_env1 = layer_permute(output_env1)
    output_env2 = layer_permute(output_env2)

    layer_reshape = tf.keras.layers.Reshape((tf.keras.backend.int_shape(output_env1)[1],
                                             tf.keras.backend.int_shape(output_env1)[2] *
                                             tf.keras.backend.int_shape(output_env1)[3]))
    output_env1 = layer_reshape(output_env1)  # size = (210,32)
    output_env2 = layer_reshape(output_env2)


    output_mel1 = layer_permute(output_mel1)
    output_mel2 = layer_permute(output_mel2)

    layer_reshape = tf.keras.layers.Reshape((tf.keras.backend.int_shape(output_mel1)[1],
                                             tf.keras.backend.int_shape(output_mel1)[2] *
                                             tf.keras.backend.int_shape(output_mel1)[3]))
    output_mel1 = layer_reshape(output_mel1)
    output_mel2 = layer_reshape(output_mel2)

    output_spch1 = tf.keras.layers.Concatenate()([output_env1, output_mel1])
    output_spch2 = tf.keras.layers.Concatenate()([output_env2, output_mel2])


    # lstm_spch = tf.keras.layers.LSTM(units_lstm, return_sequences=True, activation= fun_act)
    lstm_spch = tf.keras.layers.CuDNNLSTM(units_lstm, return_sequences=True)
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

    model = tf.keras.Model(inputs=[input_eeg, env1, env2, mel1, mel2], outputs=[y_out, cos_scores_sig])

    return model