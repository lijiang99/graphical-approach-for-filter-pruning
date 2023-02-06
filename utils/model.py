from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, Activation, BatchNormalization
from tensorflow.keras.layers import Dropout, MaxPooling2D, AveragePooling2D, Flatten
from tensorflow.keras.layers import Dense, Concatenate, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model

# vgg architecture
def stack_layer(inputs, num_filters=64, drop_rate=None):
    conv = Conv2D(num_filters, (3,3), padding="same", kernel_regularizer=l2(5e-4))
    x = conv(inputs)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(drop_rate)(x) if drop_rate is not None else MaxPooling2D(pool_size=(2,2))(x)
    return x

def vgg16(input_shape, num_classes=10, mask_nums=None):
    inputs, idx = Input(shape=input_shape), 0
    mask_nums = mask_nums if mask_nums is not None else [0]*13
    
    x = stack_layer(inputs=inputs, num_filters=64-mask_nums[idx], drop_rate=0.3)
    idx += 1
    x = stack_layer(inputs=x, num_filters=64-mask_nums[idx], drop_rate=None)
    idx += 1
    
    x = stack_layer(inputs=x, num_filters=128-mask_nums[idx], drop_rate=0.4)
    idx += 1
    x = stack_layer(inputs=x, num_filters=128-mask_nums[idx], drop_rate=None)
    idx += 1
    
    x = stack_layer(inputs=x, num_filters=256-mask_nums[idx], drop_rate=0.4)
    idx += 1
    x = stack_layer(inputs=x, num_filters=256-mask_nums[idx], drop_rate=0.4)
    idx += 1
    x = stack_layer(inputs=x, num_filters=256-mask_nums[idx], drop_rate=None)
    idx += 1
    
    x = stack_layer(inputs=x, num_filters=512-mask_nums[idx], drop_rate=0.4)
    idx += 1
    x = stack_layer(inputs=x, num_filters=512-mask_nums[idx], drop_rate=0.4)
    idx += 1
    x = stack_layer(inputs=x, num_filters=512-mask_nums[idx], drop_rate=None)
    idx += 1
    
    x = stack_layer(inputs=x, num_filters=512-mask_nums[idx], drop_rate=0.4)
    idx += 1
    x = stack_layer(inputs=x, num_filters=512-mask_nums[idx], drop_rate=0.4)
    idx += 1
    x = stack_layer(inputs=x, num_filters=512-mask_nums[idx], drop_rate=None)
    
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(512-mask_nums[idx], kernel_regularizer=l2(5e-4))(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# resnet architecture
def resnet_layer(inputs, num_filters=16, kernel_size=3, strides=1,
                 activation="relu", batch_normalization=True, conv_first=True):
    conv = Conv2D(num_filters, kernel_size=kernel_size, strides=strides, padding="same",
                  kernel_initializer="he_normal", kernel_regularizer=l2(1e-4))
    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

def resnet(input_shape, depth, num_classes=10, mask_nums=None):
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)
    mask_nums = mask_nums if mask_nums is not None else [0]*(depth+1)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs, num_filters=num_filters-mask_nums[0])
    idx = 1
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:
                strides = 2
            y = resnet_layer(inputs=x, num_filters=num_filters-mask_nums[idx], strides=strides)
            idx += 1
            y = resnet_layer(inputs=y, num_filters=num_filters-mask_nums[idx], activation=None)
            idx += 1
            if stack > 0 and res_block == 0:
                x = resnet_layer(inputs=x, num_filters=num_filters-mask_nums[idx], kernel_size=1,
                                 strides=strides, activation=None, batch_normalization=False)
                idx += 1
            x = keras.layers.add([x, y])
            x = Activation("relu")(x)
        num_filters *= 2

    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes, activation="softmax", kernel_initializer="he_normal")(y)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# densenet architecture
def conv_block(x, num_filters, drop_rate, dilate_rate):
    x = BatchNormalization(gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))(x)
    x = Activation("relu")(x)
    x = Conv2D(num_filters, (3,3), padding="same", dilation_rate=dilate_rate,
               kernel_initializer="he_uniform", kernel_regularizer=l2(1e-4), use_bias=False)(x)
    x = Dropout(rate=drop_rate)(x)
    return x

def transition_layer(x, num_filters, dilate_rate, drop_rate, compression):
    x = BatchNormalization(gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))(x)
    x = Activation("relu")(x)
    if compression != 1:
        x = Conv2D(int(num_filters*compression), (1,1), padding="same", dilation_rate=dilate_rate,
                   kernel_initializer="he_uniform", use_bias=False, kernel_regularizer=l2(1e-4))(x)
    else:
        x = Conv2D(num_filters, (1,1), padding="same", dilation_rate=dilate_rate,
                   kernel_initializer="he_uniform", use_bias=False, kernel_regularizer=l2(1e-4))(x)
    
    x = Dropout(rate=drop_rate)(x)
    x = AveragePooling2D((2,2), strides=(2,2))(x)
    return x

def dense_block(x, num_layers, num_filters, grow_rate, drop_rate, dilate_rate, mask_nums):
    concat_layers = [x]
    for i in range(num_layers):
        conv_b = conv_block(x, grow_rate-mask_nums[i], drop_rate, dilate_rate)
        concat_layers.append(conv_b)
        x = Concatenate(axis=-1)(concat_layers)
        num_filters += grow_rate
    return x, num_filters

def densenet(input_shape, num_layers=12, dilate_rate=1, grow_rate=12, num_filters=16,
             num_classes=10, drop_rate=0.2, num_blocks=3, compression=1, mask_nums=None):
    inputs = Input(shape=input_shape)
    mask_nums = mask_nums if mask_nums is not None else [0]*((num_layers+1)*num_blocks)
    
    x = Conv2D(num_filters-mask_nums[0], (3,3), padding="same", dilation_rate=dilate_rate,
               kernel_initializer="he_uniform", kernel_regularizer=l2(1e-4), use_bias=False)(inputs)
    
    idx = 1
    for i in range(0, num_blocks-1):
        x, num_filters = dense_block(x=x, num_layers=num_layers, num_filters=num_filters, grow_rate=grow_rate,
                                     drop_rate=drop_rate, dilate_rate=dilate_rate, mask_nums=mask_nums[idx:idx+num_layers])
        x = transition_layer(x=x, num_filters=num_filters-mask_nums[idx+num_layers],
                             dilate_rate=dilate_rate, drop_rate=drop_rate, compression=compression)
        idx += (num_layers+1)
    
    x, num_filters = dense_block(x=x, num_layers=num_layers, num_filters=num_filters, grow_rate=grow_rate,
                                 drop_rate=drop_rate, dilate_rate=dilate_rate, mask_nums=mask_nums[idx:])
    
    x = BatchNormalization(gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))(x)
    x = Activation("relu")(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes, activation="softmax", kernel_regularizer=l2(1e-4),
              bias_regularizer=l2(1e-4))(x)
    model = Model(inputs=inputs, outputs=[x])
    return model