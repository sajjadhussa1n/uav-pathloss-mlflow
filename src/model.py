import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import concatenate, Conv2D, MaxPooling2D, BatchNormalization, Activation, Dropout, Input, Add, GlobalAveragePooling2D, Reshape, UpSampling2D, Conv2DTranspose, Concatenate, Multiply
from tensorflow.keras import layers, Model

def variable_conv_module(x, filters):

    if filters < 64:
        f1, f2, f3, f4, f5, d1, d2, d3, d4 = filters, filters, filters, filters, filters*2, 1, 1, 1, 1
    elif filters < 256:
        f1, f2, f3, f4, f5, d1, d2, d3, d4 = filters, filters, filters//2, filters//4, filters*2, 1, 1, 1, 1
    else:
        f1, f2, f3, f4, f5, d1, d2, d3, d4 = filters, filters, filters//4, filters//4, filters*2, 1, 2, 1, 2

    # Branch 1: 1x1 convolution
    branch1 = Conv2D(f1, (1, 1), padding='same')(x)
    branch1 = BatchNormalization()(branch1)
    branch1 = Activation('relu')(branch1)

    # Branch 1: 3x3 convolution
    branch2 = Conv2D(f2, (3, 3), dilation_rate=d1, padding='same')(x)
    branch2 = BatchNormalization()(branch2)
    branch2 = Activation('relu')(branch2)
    branch2 = Conv2D(f2, (3, 3), dilation_rate=d2, padding='same')(branch2)  # Second Conv2D
    branch2 = BatchNormalization()(branch2)
    branch2 = Activation('relu')(branch2)  # Activation added here

    # Branch 3: 5x5 convolution
    branch3 = Conv2D(f3, (5, 5), dilation_rate=d3, padding='same')(x)
    branch3 = BatchNormalization()(branch3)
    branch3 = Activation('relu')(branch3)
    branch3 = Conv2D(f3, (5, 5), dilation_rate=d4, padding='same')(branch3)  # Second Conv2D
    branch3 = BatchNormalization()(branch3)
    branch3 = Activation('relu')(branch3)  # Activation added here
    if filters < 64:
        # Branch 4: 7x7 convolution
        branch4 = Conv2D(f4, (7, 7), padding='same')(x)
        branch4 = BatchNormalization()(branch4)
        branch4 = Activation('relu')(branch4)
        merged = layers.Concatenate()([branch1, branch2, branch3, branch4])
    else:
        merged = layers.Concatenate()([branch1, branch2, branch3])

    # Feature Fusion by compressing channels: 2.75*filters → filters
    merged = Conv2D(f5, (1, 1), padding='same')(merged)
    merged = BatchNormalization()(merged)
    merged = Activation('relu')(merged)

    return merged


def conv_block(inputs, num_filters):
    """
    A block consisting of two 3x3 convolutional layers
    """
    x = layers.Conv2D(num_filters, (3,3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(num_filters, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def encoder_block(inputs, filters):
    x = variable_conv_module(inputs, filters)
    p = MaxPooling2D(2)(x)
    return x, p


def decoder_block(inputs, skip_features, num_filters):
    x = layers.Conv2DTranspose(num_filters, 3, strides=2, padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = layers.Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

## ASPP Bottleneck
def aspp_block(x, filters=512):
    input_filters = x.shape[-1]
    # 1x1 convolution
    conv1x1 = Conv2D(filters, 1, padding='same')(x)

    # Dilated convolutions
    conv3x3_d6 = Conv2D(filters, 3, padding='same', dilation_rate=1)(x)
    conv3x3_d12 = Conv2D(filters, 3, padding='same', dilation_rate=2)(x)
    conv3x3_d18 = Conv2D(filters, 3, padding='same', dilation_rate=4)(x)

    # Global average pooling
    pool = GlobalAveragePooling2D()(x)
    pool = Reshape((1,1,input_filters))(pool)
    pool = Conv2D(filters, 1, padding='same', use_bias=False)(pool)
    pool = BatchNormalization()(pool)
    pool = Activation('relu')(pool)
    pool = UpSampling2D(size=(x.shape[1], x.shape[2]), interpolation='bilinear')(pool)

    # Concatenate and project
    concat = Concatenate()([conv1x1, conv3x3_d6, conv3x3_d12, conv3x3_d18, pool])
    output = Conv2D(filters, 1, padding='same', use_bias=False)(concat)
    #output = BatchNormalization()(output)
    #output = Activation('relu')(output)
    #output = Dropout(0.2)(output)
    return output


def build_novel_unet(input_shape=(128, 128, 3)):
    """
    Build a U-Net model for regression.
    """
    inputs = layers.Input(input_shape)

    # Encoder
    filters = 32
    s1, p1 = encoder_block(inputs, filters)
    filters2 = 64 #tuple(x * 2 for x in filters)
    s2, p2 = encoder_block(p1, filters2)
    filters3 = 128 #tuple(x * 2 for x in filters2)
    s3, p3 = encoder_block(p2, filters3)
    filters4 = 256 #tuple(x * 2 for x in filters3)
    s4, p4 = encoder_block(p3, filters4)

    # Bottleneck
    b1 = aspp_block(p4, 512)

    # Decoder (apply attention to deeper layers only)
    d1 = decoder_block(b1, s4, 256)  # Deepest skip (s4)
    d2 = decoder_block(d1, s3, 128)    # Mid-depth (s3)
    d3 = decoder_block(d2, s2, 64)   # Shallow (s2)
    d4 = decoder_block(d3, s1, 32)   # Shallowest (s1)


    # Output layer
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(d4)

    # Compile the model
    model = Model(inputs, outputs, name="U-Net")
    return model
