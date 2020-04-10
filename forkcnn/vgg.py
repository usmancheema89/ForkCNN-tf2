'''VGGFace models for Keras.

# Notes:
- Resnet50 and VGG16  are modified architectures from Keras Application folder. [Keras](https://keras.io)

- Squeeze and excitation block is taken from  [Squeeze and Excitation Networks in
 Keras](https://github.com/titu1994/keras-squeeze-excite-network) and modified.

'''

from keras.layers import Flatten, Dense, Input, Activation, Conv2D, MaxPooling2D, concatenate, add, Dropout
# from keras_applications.imagenet_utils import _obtain_input_shape
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras_vggface import utils
from keras.engine.topology import get_source_inputs
import warnings
from keras.models import Model


def bottom(img_input, name):  # 0.3077
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1_' + name)(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2_' + name)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1_' + name)(x)
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1_' + name)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2_' + name)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2_' + name)(x)
    return x


def mid(x, name):  # 0.1538 %     Approx. 50% of network (45% :D)
    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1_' + name)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2_' + name)(x)
    return x


def midtop(x, name):  # 0.1538     Approx. 65% of network
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3_' + name)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3_' + name)(x)
    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1_' + name)(x)
    return x


def top(x, name):  # 0.3846     100%
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2_' + name)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3_' + name)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool4_' + name)(x)
    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3')(x)
    z = MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(x)
    return z


def VGG16_vanilla(input_image_1):
    # img_input = Input(shape=input_shape)
    # Normal VGG (single stream)
    print("Using single stream VGG16")
    output = top(midtop(mid(bottom(input_image_1, 'visible_stream'))))
    return output


def VGG16_two_stream_30(input_image_1, input_image_2, merge):
    x_1 = bottom(input_image_1, 'visible_stream')
    x_2 = bottom(input_image_2, 'thermal_stream')
    x = combine_stream(x_1, x_2, merge)
    print("Using two streams (VGG16) with merging strings after 30% of conv layers")

    # Network is single streamed
    return top(midtop(mid(x, 'merged'), 'merged'), 'merged')


def VGG16_two_stream_50(input_image_1, input_image_2, merge):
    x_1 = mid(bottom(input_image_1, 'visible_stream'), 'visible_stream')
    x_2 = mid(bottom(input_image_2, 'thermal_stream'), 'thermal_stream')
    x = combine_stream(x_1, x_2, merge)
    print("Using two streams (VGG16) with merging strings after 50% of conv layers")

    # Network is single streamed
    return top(midtop(x, 'merged'), 'merged')


def VGG16_two_stream_70(input_image_1, input_image_2, merge):
    x_1 = midtop(mid(bottom(input_image_1, 'visible_stream'), 'visible_stream'), 'visible_stream')
    x_2 = midtop(mid(bottom(input_image_2, 'thermal_stream'), 'thermal_stream'), 'thermal_stream')
    x = combine_stream(x_1, x_2, merge)
    # Network is single streamed
    print("Using two streams (VGG16) with merging strings after 70% of conv layers")
    return top(x, 'merged')


def combine_stream(x_1, x_2, merge):
    if merge == "concatenate":
        return concatenate([x_1, x_2])
    if merge == "addition":
        return add(x_1, x_2)


def VGG16(input_shape, include_top, input_1_tensor, input_2_tensor, stream, merge_style,
          merge_point, pooling, weights, classes):
    input_image_1 = Input(shape=input_shape)
    input_image_2 = Input(shape=input_shape)
    if stream == 1:
        output = VGG16_vanilla(input_image_1)
    if stream == 2:
        if merge_point == 30:
            output = VGG16_two_stream_30(input_image_1, input_image_2, merge_style)

        if merge_point == 50:
            output = VGG16_two_stream_50(input_image_1, input_image_2, merge_style)

        if merge_point == 70:
            output = VGG16_two_stream_70(input_image_1, input_image_2, merge_style)

    output = Flatten(name='flatten')(output)
    output = Dense(4096, name='fc6')(output)
    output = Activation('relu', name='fc6/relu')(output)
    output = Dropout(0.5)(output)
    # x = Dense(4096, name='fc7')(x)
    # x = Activation('relu', name='fc7/relu')(x)
    output = Dense(classes, name='fc8')(output)
    output = Activation('softmax', name='fc8/softmax')(output)

    # if include_top:
    #     # Classification block
    #     x = Flatten(name='flatten')(x)
    #     x = Dense(4096, name='fc6')(x)
    #     x = Activation('relu', name='fc6/relu')(x)
    #     x = Dense(4096, name='fc7')(x)
    #     x = Activation('relu', name='fc7/relu')(x)
    #     x = Dense(classes, name='fc8')(x)
    #     x = Activation('softmax', name='fc8/softmax')(x)
    # else:
    #     if pooling == 'avg':
    #         x = GlobalAveragePooling2D()(x)
    #     elif pooling == 'max':
    #         x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_1_tensor is not None:
        inputs = get_source_inputs(input_1_tensor)
    else:

        inputs = [input_image_1, input_image_2]

        # Create model.
    model = Model(inputs, output, name='vggface_vgg16_2stream')  # load weights
    if weights == 'vggface':  # TODO How to load weights from trained model to a two-stream network.
        if include_top:  # TODO One idea is to duplicate the single-stream network weights to both streams.
            weights_path = get_file('rcmalli_vggface_tf_vgg16.h5',
                                    utils.
                                    VGG16_WEIGHTS_PATH,
                                    cache_subdir=utils.VGGFACE_DIR)
        else:
            weights_path = get_file('rcmalli_vggface_tf_notop_vgg16.h5',
                                    utils.VGG16_WEIGHTS_PATH_NO_TOP,
                                    cache_subdir=utils.VGGFACE_DIR)
        model.load_weights(weights_path, by_name=True)
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)

        if K.image_data_format() == 'channels_first':
            if include_top:
                maxpool = model.get_layer(name='pool5')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='fc6')
                layer_utils.convert_dense_weights_data_format(dense, shape,
                                                              'channels_first')

            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
    return model
