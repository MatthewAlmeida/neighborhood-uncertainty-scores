"""
dense_net.py

Adaptation of keras library code for 
densenet, adapted for 1D convolutions.
Adding it here in case we want to use
a large convolutional model.

"""

from tensorflow.keras import (
    layers,
    models, 
    backend,
    optimizers,
    regularizers
)

def dense_block(x, blocks, growth_rate, dropout_rate, mid_csv, reg, name):
    """A dense block.
    # Arguments
        x: input tensor.
        blocks: integer, the number of building blocks.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    for i in range(blocks):
        x = conv_block(x, growth_rate, dropout_rate, mid_csv, reg, name=name + '_block' + str(i + 1))
    return x


def transition_block(x, reduction, reg, name):
    """A transition block.
    # Arguments
        x: input tensor.
        reduction: float, compression rate at transition layers.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    bn_axis = 2
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_bn')(x)
    x = layers.Activation('relu', name=name + '_relu')(x)
    x = layers.Conv1D(int(backend.int_shape(x)[bn_axis] * reduction), 1,
                      use_bias=False, kernel_regularizer=reg,
                      name=name + '_conv')(x)
    x = layers.AveragePooling1D(2, strides=2, name=name + '_pool')(x)
    return x


def conv_block(x, growth_rate, dropout_rate, mid_cvs, reg, name):
    """A building block for a dense block.
    # Arguments
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        name: string, block label.
    # Returns
        Output tensor for the block.
    """
    bn_axis = 2

    x1 = layers.BatchNormalization(axis=bn_axis,
                                   epsilon=1.001e-5,
                                   name=name + '_0_bn')(x)

    x1 = layers.Activation('relu', name=name + '_0_relu')(x1)

    x1 = layers.Conv1D(4 * growth_rate, 1,
                       use_bias=False,
                       kernel_regularizer=reg,
                       name=name + '_1_conv')(x1)

    x1 = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                   name=name + '_1_bn')(x1)

    x1 = layers.Activation('relu', name=name + '_1_relu')(x1)

    x1 = layers.Conv1D(growth_rate, mid_cvs,
                       padding='same', use_bias=False,
                       kernel_regularizer=reg,
                       name=name + '_2_conv')(x1)
    x1 = layers.Dropout(dropout_rate)(x1)

    x = layers.Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])

    return x

def build_dense_net(
    blocks,
    output_dim,
    input_shape=None,
    growth_rate=32,
    lr=0.01, dpt=0.5, mid_dpt=0.0,
    cvf=128, cvs = 7, mid_cvs=3,
    use_l2=1, l2_val=0.001
):
    # In 2D convolution with keras, we pass
    # the channels axis here. The equivalent
    # axis for us is the second one, which 
    # is the number of features describing each 
    # time step

    bn_axis = 2
   
    # If we want to use an l2 penalty, reg is 
    # initialized to the usual keras l2 regularizer.
    # If not, we use a lambda function that always
    # returns 0 (no penalty)

    if use_l2 == 1:
        reg = regularizers.l2(l2_val)
    else:
        reg = lambda weight_matrix: 0.0

    input_tensor = layers.Input(shape=input_shape)

    x = layers.Conv1D(
            cvf, cvs, strides=1, use_bias=False, 
            kernel_regularizer=reg,
            padding = "same", name='conv1/conv'
        )(input_tensor)

    x = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name='conv1/bn'
        )(x)

    x = layers.Activation('relu', name='conv1/relu')(x)
    x = layers.AveragePooling1D(3, name='pool1')(x)

    for i, val in enumerate(blocks):
        x = dense_block(x, val, growth_rate, mid_dpt, mid_cvs, reg, name=f'dense_block_{i}')
        x = transition_block(x, 0.5, reg, name=f'transition_block_{i}')

    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name='bn')(x)
    x = layers.Dropout(dpt)(x)
    x = layers.Activation('relu', name='relu')(x)
    x = layers.GlobalAveragePooling1D(name='avg_pool')(x)
    
    if output_dim > 1:
        x = layers.Dense(output_dim, activation='softmax', name='fc1000')(x)
    else:
        x = layers.Dense(output_dim, activation='sigmoid', name='fc1000')(x)

    model = models.Model(
        input_tensor,
        x,
        name="densenet"
    )
    
    return model
