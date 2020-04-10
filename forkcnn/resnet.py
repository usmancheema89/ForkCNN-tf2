'''VGGFace models for Keras.

# Notes:
- Resnet50 and VGG16  are modified architectures from Keras Application folder. [Keras](https://keras.io)

- Squeeze and excitation block is taken from  [Squeeze and Excitation Networks in
 Keras](https://github.com/titu1994/keras-squeeze-excite-network) and modified.

'''

from keras.layers import Flatten, Dense, Input, GlobalAveragePooling2D, \
    GlobalMaxPooling2D, Activation, Conv2D, MaxPooling2D, BatchNormalization, \
    AveragePooling2D, Reshape, Permute, multiply, concatenate, add, Dropout
from keras_applications.imagenet_utils import _obtain_input_shape
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras_vggface import utils
from keras.engine.topology import get_source_inputs
import warnings
from keras.models import Model
from keras import layers


def combine_stream(x_1, x_2, merge):
    if merge == "concatenate":
        return concatenate([x_1, x_2], name="STREAM_MERGE_CONCAT")
    if merge == "addition":
        return add([x_1, x_2], name="STREAM_MERGE_ADD")


def bottom(image_input, bn_axis, name):
    x = Conv2D(64, (7, 7), use_bias=False, strides=(2, 2), padding='same',
               name='conv1/7x7_s2_' + name)(image_input)
    x = BatchNormalization(axis=bn_axis, name='conv1/7x7_s2/bn_' + name)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = resnet_conv_block(x, 3, [64, 64, 256], stage=2, block=1, strides=(1, 1), name=name)
    x = resnet_identity_block(x, 3, [64, 64, 256], stage=2, block=2, name=name)
    x = resnet_identity_block(x, 3, [64, 64, 256], stage=2, block=3, name=name)
    x = resnet_conv_block(x, 3, [128, 128, 512], stage=3, block=1, name=name)
    return x


def mid(x, name):
    filter3 = K.int_shape(x)[3]
    x = resnet_identity_block(x, 3, [128, 128, filter3], stage=3, block=2, name=name)
    x = resnet_identity_block(x, 3, [128, 128, filter3], stage=3, block=3, name=name)
    x = resnet_identity_block(x, 3, [128, 128, filter3], stage=3, block=4, name=name)
    return x


def midtop(x, name):
    filter3 = K.int_shape(x)[3]
    x = resnet_conv_block(x, 3, [256, 256, filter3], stage=4, block=1, name=name)
    x = resnet_identity_block(x, 3, [256, 256, filter3], stage=4, block=2, name=name)
    x = resnet_identity_block(x, 3, [256, 256, filter3], stage=4, block=3, name=name)
    return x


def top(x, name):
    filter3 = K.int_shape(x)[3]
    x = resnet_identity_block(x, 3, [256, 256, filter3], stage=4, block=4, name=name)
    x = resnet_identity_block(x, 3, [256, 256, filter3], stage=4, block=5, name=name)
    x = resnet_identity_block(x, 3, [256, 256, filter3], stage=4, block=6, name=name)

    x = resnet_conv_block(x, 3, [512, 512, filter3], stage=5, block=1, name=name)
    x = resnet_identity_block(x, 3, [512, 512, filter3], stage=5, block=2, name=name)
    x = resnet_identity_block(x, 3, [512, 512, filter3], stage=5, block=3, name=name)

    x = AveragePooling2D((7, 7), name='avg_pool')(x)
    return x


def resnet_identity_block(input_tensor, kernel_size, filters, stage, block,
                          bias=False, name=None):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv1_reduce_name = 'conv' + str(stage) + "_" + str(block) + "_1x1_reduce_" + name
    conv1_increase_name = 'conv' + str(stage) + "_" + str(
        block) + "_1x1_increase_" + name
    conv3_name = 'conv' + str(stage) + "_" + str(block) + "_3x3_" + name

    x = Conv2D(filters1, (1, 1), use_bias=bias, name=conv1_reduce_name)(
        input_tensor)
    x = BatchNormalization(axis=bn_axis, name=conv1_reduce_name + "/bn")(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, use_bias=bias,
               padding='same', name=conv3_name)(x)
    x = BatchNormalization(axis=bn_axis, name=conv3_name + "/bn")(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), use_bias=bias, name=conv1_increase_name)(x)
    x = BatchNormalization(axis=bn_axis, name=conv1_increase_name + "/bn")(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def resnet_conv_block(input_tensor, kernel_size, filters, stage, block,
                      strides=(2, 2), bias=False, name=None):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv1_reduce_name = 'conv' + str(stage) + "_" + str(block) + "_1x1_reduce_" + name
    conv1_increase_name = 'conv' + str(stage) + "_" + str(
        block) + "_1x1_increase_" + name
    conv1_proj_name = 'conv' + str(stage) + "_" + str(block) + "_1x1_proj_" + name
    conv3_name = 'conv' + str(stage) + "_" + str(block) + "_3x3_" + name

    x = Conv2D(filters1, (1, 1), strides=strides, use_bias=bias,
               name=conv1_reduce_name)(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=conv1_reduce_name + "/bn")(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', use_bias=bias,
               name=conv3_name)(x)
    x = BatchNormalization(axis=bn_axis, name=conv3_name + "/bn")(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv1_increase_name, use_bias=bias)(x)
    x = BatchNormalization(axis=bn_axis, name=conv1_increase_name + "/bn")(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides, use_bias=bias,
                      name=conv1_proj_name)(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=conv1_proj_name + "/bn")(
        shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def RESNET50_vanilla(input_image_1, bn_axis):
    print("Using single stream VGG16")
    output = bottom(input_image_1, bn_axis, 'single_stream')
    output = top(midtop(mid(output, 'single_stream'), 'single_stream'), 'single_stream')
    return output


def RESNET50_two_stream_30(input_image_1, input_image_2, bn_axis, merge_style):
    x_1 = bottom(input_image_1, bn_axis, 'visible_stream')
    x_2 = bottom(input_image_2, bn_axis, 'thermal_stream')
    output = combine_stream(x_1, x_2, merge_style)
    output = mid(output, 'merged')
    output = midtop(output, 'merged')
    return top(output, 'merged')
    # return top(midtop(mid(output)))


def RESNET50_two_stream_50(input_image_1, input_image_2, bn_axis, merge_style):
    x_1 = mid(bottom(input_image_1, bn_axis, 'visible_stream'), name='visible_stream')
    x_2 = mid(bottom(input_image_2, bn_axis, 'thermal_stream'), name='thermal_stream')
    output = combine_stream(x_1, x_2, merge_style)
    return top(midtop(output, 'merged'), 'merged')


def RESNET50_two_stream_70(input_image_1, input_image_2, bn_axis, merge_style):
    x_1 = midtop(mid(bottom(input_image_1, bn_axis, 'visible_stream'), name='visible_stream'), name='visible_stream')
    x_2 = midtop(mid(bottom(input_image_2, bn_axis, 'thermal_stream'), name='thermal_stream'), name='thermal_stream')
    output = combine_stream(x_1, x_2, merge_style)
    return top(output, 'merged')


def RESNET50(input_shape, include_top, input_1_tensor, input_2_tensor, stream, merge_style, merge_point, pooling,
             weights, classes):
    # input_shape = _obtain_input_shape(input_shape,
    #                                   default_size=224,
    #                                   min_size=32,
    #                                   data_format=K.image_data_format(),
    #                                   require_flatten=include_top,
    #                                   weights=weights)

    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    input_image_1 = Input(shape=input_shape)
    input_image_2 = Input(shape=input_shape)
    if stream == 1:
        inputs = input_image_1
        output = RESNET50_vanilla(input_image_1, bn_axis)

    if stream == 2:
        inputs = [input_image_1, input_image_2]
        if merge_point == 30:
            output = RESNET50_two_stream_30(input_image_1, input_image_2, bn_axis, merge_style)

        if merge_point == 50:
            output = RESNET50_two_stream_50(input_image_1, input_image_2, bn_axis, merge_style)

        if merge_point == 70:
            output = RESNET50_two_stream_70(input_image_1, input_image_2, bn_axis, merge_style)
    if include_top:
        output = Flatten()(output)
        output = Dense(classes, activation='softmax', name='classifier')(output)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(output)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(output)

    # Create model.
    model = Model(inputs, output, name='vggface_resnet50')

    # load weights
    if weights == 'vggface':
        if include_top:
            weights_path = get_file('rcmalli_vggface_tf_resnet50.h5',
                                    utils.RESNET50_WEIGHTS_PATH,
                                    cache_subdir=utils.VGGFACE_DIR)
        else:
            weights_path = get_file('rcmalli_vggface_tf_notop_resnet50.h5',
                                    utils.RESNET50_WEIGHTS_PATH_NO_TOP,
                                    cache_subdir=utils.VGGFACE_DIR)
        model.load_weights(weights_path)
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)
            if include_top:
                maxpool = model.get_layer(name='avg_pool')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='classifier')
                layer_utils.convert_dense_weights_data_format(dense, shape,
                                                              'channels_first')

        if K.image_data_format() == 'channels_first' and K.backend() == 'tensorflow':
            warnings.warn('You are using the TensorFlow backend, yet you '
                          'are using the Theano '
                          'image data format convention '
                          '(`image_data_format="channels_first"`). '
                          'For best performance, set '
                          '`image_data_format="channels_last"` in '
                          'your Keras config '
                          'at ~/.keras/keras.json.')
    elif weights is not None:
        model.load_weights(weights)

    return model
