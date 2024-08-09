# -*- coding: utf-8 -*-


from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras.layers import Conv1D ,DepthwiseConv1D,MaxPooling1D



def get_top(x_input):
    """Block top operations
    This functions apply Batch Normalization and Leaky ReLU activation to the input.
    # Arguments:
        x_input: Tensor, input to apply BN and activation  to.
    # Returns:
        Output tensor
    """
    
    x = tf.keras.layers.BatchNormalization()(x_input)
    x = tf.keras.layers.LeakyReLU()(x)
    return x

def get_block(x_input, input_channels, output_channels):
    """MBConv block
    This function defines a mobile Inverted Residual Bottleneck block with BN and Leaky ReLU
    # Arguments
        x_input: Tensor, input tensor of conv layer.
        input_channels: Integer, the dimentionality of the input space.
        output_channels: Integer, the dimensionality of the output space.
            
    # Returns
        Output tensor.
    """

    x = tf.keras.layers.Conv1D(input_channels, kernel_size=2, padding='same')(x_input)
    x = get_top(x)
    x = tf.keras.layers.DepthwiseConv1D(kernel_size=(2), padding='same')(x)
    x = get_top(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=(2), strides=(2))(x)
    x = tf.keras.layers.DepthwiseConv1D(kernel_size=(2), padding='same')(x)
    x = get_top(x)
    x = tf.keras.layers.Conv1D(output_channels, kernel_size=(2), strides=(2), padding='same')(x)
    return x


def EffNet(input_shape, num_classes, plot_model=False):
    """EffNet
    This function defines a EfficientNet architecture.
    # Arguments
        input_shape: An integer or tuple/list of 1 integers, shape
            of input tensor.
        num_classes: Integer, number of classes.

    # Returns
        EfficientNet model.
    """
    x_input = tf.keras.layers.Input(shape=input_shape)
    x = get_block(x_input, 32, 64)
    x = get_block(x, 64, 128)
    x = get_block(x, 128, 256)
    
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=x_input, outputs=x)
    
    model.compile(
        optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"]
    )
    
    return model
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
   


